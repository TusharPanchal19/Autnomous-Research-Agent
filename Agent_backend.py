"""
Autonomous Research Agent - Single File Backend

Pipeline:
Question
 -> Planning
 -> Tavily Search
 -> Web Extraction
 -> Chunking
 -> Chroma Retriever
 -> Evidence Analyst
 -> Critic
 -> Human Approval
 -> (Research More -> Refine Query -> Tavily -> ...) OR
 -> Markdown + PDF Report

MCP is intentionally not included.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from pathlib import Path
from functools import lru_cache
from typing import Literal, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from tavily import TavilyClient

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import interrupt, Command

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from xml.sax.saxutils import escape


# ============================================================
# 1. CONFIGURATION
# ============================================================

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
REPORT_DIR = BASE_DIR / "reports"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

CHROMA_COLLECTION = os.getenv(
    "CHROMA_COLLECTION",
    "autonomous_research_documents",
)

RETRIEVER_K = int(os.getenv("RETRIEVER_K", "8"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))


# ============================================================
# RUNTIME PROGRESS
# ============================================================

_progress_lock = threading.Lock()
_progress = {}


def _new_progress(thread_id: str):
    with _progress_lock:
        _progress[thread_id] = {
            "stage": "starting",
            "status": "running",
            "message": "Starting research...",
            "detail": "",
            "history": [],
        }


def _set_progress(
    thread_id: str,
    stage: str,
    status: str,
    message: str,
    detail: str = "",
):
    event = {
        "stage": stage,
        "status": status,
        "message": message,
        "detail": detail,
    }

    with _progress_lock:
        current = _progress.setdefault(
            thread_id,
            {
                "stage": stage,
                "status": status,
                "message": message,
                "detail": detail,
                "history": [],
            },
        )

        current.update(event)

        if status in {"done", "waiting", "failed", "stopped"}:
            history = current.setdefault("history", [])
            history.append(event)
            current["history"] = history[-20:]


def get_progress(thread_id: str) -> dict:
    with _progress_lock:
        value = _progress.get(thread_id)

        if not value:
            return {
                "stage": "unknown",
                "status": "idle",
                "message": "No active research job.",
                "detail": "",
                "history": [],
            }

        return {
            "stage": value.get("stage", "unknown"),
            "status": value.get("status", "idle"),
            "message": value.get("message", ""),
            "detail": value.get("detail", ""),
            "history": list(value.get("history", [])),
        }


def _stage_start(thread_id, stage, message, detail=""):
    _set_progress(thread_id, stage, "running", message, detail)


def _stage_done(thread_id, stage, message, detail=""):
    _set_progress(thread_id, stage, "done", message, detail)


def _stage_waiting(thread_id, stage, message, detail=""):
    _set_progress(thread_id, stage, "waiting", message, detail)


# ============================================================
# 2. LLM
# ============================================================

@lru_cache(maxsize=1)
def get_llm():
    return ChatOllama(
        model=OLLAMA_MODEL,
        temperature=0,
        base_url=OLLAMA_BASE_URL,
    )


# ============================================================
# 3. PYDANTIC SCHEMAS
# ============================================================

class ResearchPlan(BaseModel):
    research_questions: list[str]
    search_queries: list[str]


class ResearchFinding(BaseModel):
    claim: str
    supporting_source: str


class EvidenceAnalysis(BaseModel):
    answer: str
    findings: list[ResearchFinding]
    evidence_sufficient: bool
    limitations: list[str]


class CriticReview(BaseModel):
    approved: bool
    score: int = Field(ge=0, le=10)
    unsupported_claims: list[str]
    missing_information: list[str]
    contradictions: list[str]
    feedback: str


class QueryRefinement(BaseModel):
    search_queries: list[str]
    reasoning: str


class ResearchReport(BaseModel):
    title: str
    executive_summary: str
    key_findings: list[str]
    detailed_analysis: str
    limitations: list[str]
    conclusion: str


class ResearchState(TypedDict, total=False):
    question: str
    thread_id: str
    search_queries: list[str]
    refinement_reasoning: str

    retrieved_docs: list[dict]
    formatted_evidence: str
    source_count: int
    chunk_count: int
    retrieved_count: int

    # Keep checkpointed state JSON/MsgPack serializable.
    # Pydantic objects are reconstructed inside nodes when needed.
    plan: dict
    analysis: dict
    critic_review: dict

    retry_count: int
    max_retries: int

    human_decision: str

    markdown_report: str
    markdown_path: str
    pdf_path: str


# ============================================================
# 4. TAVILY SEARCH
# ============================================================

@lru_cache(maxsize=1)
def get_tavily_client():
    if not TAVILY_API_KEY:
        raise RuntimeError(
            "TAVILY_API_KEY is missing from .env"
        )
    return TavilyClient(api_key=TAVILY_API_KEY)


def tavily_search(query: str, max_results: int = 5) -> list[dict]:
    response = get_tavily_client().search(
        query=query,
        search_depth="advanced",
        max_results=max_results,
    )

    return [
        {
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "content": r.get("content", ""),
            "score": r.get("score", 0.0),
        }
        for r in response.get("results", [])
    ]


def search_queries(queries: list[str], max_results: int = 8):
    results = []

    for query in queries:
        try:
            results.extend(
                tavily_search(
                    query,
                    max_results=max_results,
                )
            )
        except Exception as exc:
            print(f"Tavily search failed: {exc}")

    return results


def deduplicate_results(results):
    seen = set()
    unique = []

    for result in results:
        url = result.get("url", "").strip()

        if url and url not in seen:
            seen.add(url)
            unique.append(result)

    return unique


# ============================================================
# 5. WEB EXTRACTION
# ============================================================

def extract_web_content(urls: list[str]):
    successful = []
    failed = []

    for url in urls:
        try:
            response = get_tavily_client().extract(
                urls=[url]
            )

            results = response.get("results", [])

            if results:
                successful.extend(results)
            else:
                failed.append(url)

        except Exception as exc:
            print(f"Extraction failed for {url}: {exc}")
            failed.append(url)

    return {
        "results": successful,
        "failed_urls": failed,
    }


# ============================================================
# 6. CHUNKING
# ============================================================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1200,
    chunk_overlap=200,
    separators=[
        "\n\n",
        "\n",
        ". ",
        " ",
        "",
    ],
)


def chunk_sources(extracted_data):
    chunks = []

    for source in extracted_data.get("results", []):

        url = source.get("url", "")

        content = (
            source.get("raw_content")
            or source.get("content")
            or ""
        )

        if not content.strip():
            continue

        source_chunks = text_splitter.split_text(content)

        for chunk_id, chunk in enumerate(source_chunks):

            chunks.append(
                {
                    "url": url,
                    "chunk_id": chunk_id,
                    "content": chunk,
                }
            )

    return chunks


# ============================================================
# 7. CHROMA + LANGCHAIN RETRIEVER
# ============================================================

@lru_cache(maxsize=1)
def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )


@lru_cache(maxsize=1)
def get_vectorstore():

    return Chroma(
        collection_name=CHROMA_COLLECTION,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )


def get_retriever(thread_id: str = ""):
    search_kwargs = {"k": RETRIEVER_K}
    if thread_id:
        search_kwargs["filter"] = {"thread_id": thread_id}

    return get_vectorstore().as_retriever(
        search_kwargs=search_kwargs
    )


def add_chunks_to_vectorstore(chunks, source_name, thread_id: str = ""):

    documents = [
        Document(
            page_content=chunk["content"],
            metadata={
                "url": chunk["url"],
                "chunk_id": chunk["chunk_id"],
                "source": source_name,
                "thread_id": thread_id,
            },
        )
        for chunk in chunks
    ]

    if documents:
        get_vectorstore().add_documents(
            documents
        )


def retrieve_evidence(question, thread_id: str = ""):

    return get_retriever(thread_id=thread_id).invoke(
        question
    )


def format_documents(documents):

    formatted = ""

    for i, doc in enumerate(documents, 1):

        formatted += f"""
================ SOURCE {i} ================

URL:
{doc.metadata.get("url", "Unknown")}

CONTENT:
{doc.page_content}

"""

    return formatted


# ============================================================
# 8. PLANNER
# ============================================================

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an expert research planner.

Convert the user's request into a research plan.

Generate:
- 3 to 8 research questions
- 3 to 8 focused search queries

Focus on factual and verifiable information.
Avoid vague queries.
Do not answer the question.
""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)

planner_chain = (
    planner_prompt
    | get_llm().with_structured_output(
        ResearchPlan
    )
)



# ============================================================
# ANALYST + CRITIC CHAINS
# ============================================================

analyst_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
            """
You are an evidence-grounded research analyst.

Answer the research question using ONLY the supplied evidence.

For every important claim:
- use evidence from the supplied sources;
- identify the supporting source URL;
- do not invent facts;
- do not invent statistics;
- do not invent dates;
- do not invent citations.

When evidence is incomplete:
- say what is known;
- identify what cannot be established;
- do not guess.

Separate:
1. Directly supported facts
2. Reasonable conclusions based on evidence
3. Unknown or insufficiently supported areas

The final answer should directly address the user's question rather
than merely summarize the sources.

Do not require every sentence to have a citation. Important factual
claims must be traceable to a source.

Return the structured output.
""",
        ),
        (
            "human",
            """Research question:
{question}

Evidence:
{evidence}""",
        ),
    ]
)

analyst_chain = analyst_prompt | get_llm().with_structured_output(
    EvidenceAnalysis
)


critic_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research quality evaluator.

Evaluate the proposed answer ONLY against the supplied evidence.

Do NOT penalize the answer simply because the evidence does not
cover every possible aspect of the topic. Judge whether the answer
adequately addresses the user's actual question.

Use this scoring rubric:

9-10:
Excellent. Claims are strongly supported, sources are relevant,
important aspects are covered, and there are no meaningful
unsupported claims or contradictions.

7-8:
Good. Most important claims are supported. There may be minor
omissions or small weaknesses, but the answer is reliable.

5-6:
Partially adequate. Some important claims lack support, or
important parts of the question are missing.

3-4:
Weak. Several important claims are unsupported or the evidence
does not adequately answer the question.

1-2:
Very poor. The answer is substantially unsupported, contradictory,
or unrelated to the supplied evidence.

IMPORTANT:
- Do not invent missing evidence.
- Do not lower the score merely because the answer is not perfect.
- Minor omissions should not cause a major score reduction.
- A limitation explicitly acknowledged by the analyst is NOT an
  unsupported claim.
- A reasonable inference is acceptable when it is clearly presented
  as an inference and supported by the evidence.
- Approve answers scoring 7 or higher unless there is a serious
  factual or evidence problem.
- Identify only meaningful problems that would affect the reliability
  of the final report.

Return:
1. score
2. approved
3. unsupported claims
4. missing important information
5. contradictions
6. concise feedback
""",
        ),
        (
            "human",
            """
Research question:
{question}

Evidence:
{evidence}

Proposed answer:
{answer}

Findings:
{findings}
""",
        ),
        (
            "human",
            """Research question:
{question}

Evidence:
{evidence}

Proposed answer:
{answer}

Findings:
{findings}""",
        ),
    ]
)

critic_chain = critic_prompt | get_llm().with_structured_output(
    CriticReview
)


def planner_node(state):

    thread_id = state["thread_id"]

    _stage_start(
        thread_id,
        "planning",
        "Building a research plan",
        "Creating focused research questions and search queries.",
    )

    plan = planner_chain.invoke(
        {"question": state["question"]}
    )

    _stage_done(
        thread_id,
        "planning",
        "Research plan created",
        f"{len(plan.research_questions)} research questions • "
        f"{len(plan.search_queries)} search queries",
    )

    return {
        "plan": plan.model_dump(),
        "search_queries": plan.search_queries,
    }

def research_node(state):

    thread_id = state["thread_id"]

    _stage_start(
        thread_id,
        "web_research",
        "Searching the web",
        "Running focused Tavily searches.",
    )

    queries = (
        state.get("search_queries")
        or [state["question"]]
    )

    results = deduplicate_results(
        search_queries(queries, max_results=5)
    )

    _set_progress(
        thread_id,
        "web_research",
        "running",
        "Collecting sources",
        f"{len(results)} unique search results found.",
    )

    urls = [
        result["url"]
        for result in results[:5]
        if result.get("url")
    ]

    _set_progress(
        thread_id,
        "web_research",
        "running",
        "Extracting source content",
        f"Opening up to {len(urls)} sources.",
    )

    extracted = extract_web_content(urls)

    successful = len(extracted.get("results", []))
    failed = len(extracted.get("failed_urls", []))

    _set_progress(
        thread_id,
        "web_research",
        "running",
        "Preparing evidence",
        f"{successful} sources extracted • {failed} extraction failures.",
    )

    chunks = chunk_sources(extracted)

    # Fallback to search result snippets if web extraction returned no chunks
    if not chunks and results:
        chunks = [
            {
                "url": r.get("url", ""),
                "chunk_id": idx,
                "content": r.get("content", "").strip(),
            }
            for idx, r in enumerate(results)
            if r.get("content", "").strip()
        ]

    _set_progress(
        thread_id,
        "web_research",
        "running",
        "Indexing evidence",
        f"{len(chunks)} text chunks prepared.",
    )

    add_chunks_to_vectorstore(
        chunks,
        "initial_research",
        thread_id=thread_id,
    )

    _set_progress(
        thread_id,
        "web_research",
        "running",
        "Retrieving relevant evidence",
        f"Selecting the top {RETRIEVER_K} chunks.",
    )

    docs = retrieve_evidence(state["question"], thread_id=thread_id)

    serializable_docs = [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata),
        }
        for doc in docs
    ]

    _stage_done(
        thread_id,
        "web_research",
        "Web research completed",
        f"{len(results)} results • {len(chunks)} chunks • "
        f"{len(docs)} evidence blocks",
    )

    return {
        "retrieved_docs": serializable_docs,
        "formatted_evidence": format_documents(docs),
        "source_count": len(results),
        "chunk_count": len(chunks),
        "retrieved_count": len(docs),
    }

def analyst_node(state):

    thread_id = state["thread_id"]

    _stage_start(
        thread_id,
        "analysis",
        "Analyzing evidence",
        "Comparing retrieved evidence against the research question.",
    )

    analysis = analyst_chain.invoke(
        {
            "question": state["question"],
            "evidence": state["formatted_evidence"],
        }
    )

    _stage_done(
        thread_id,
        "analysis",
        "Evidence analysis completed",
        f"{len(analysis.findings)} supported findings identified.",
    )

    return {
        "analysis": analysis.model_dump(),
    }

def format_findings(findings):

    return "\n".join(
        (
            f"Finding {i}: "
            f"{finding.claim}\n"
            f"Source: {finding.supporting_source}"
        )
        for i, finding in enumerate(
            findings,
            1,
        )
    )


def critic_node(state):

    thread_id = state["thread_id"]

    _stage_start(
        thread_id,
        "critic",
        "Checking research quality",
        "Looking for unsupported claims, missing information, and contradictions.",
    )

    analysis = EvidenceAnalysis.model_validate(
        state["analysis"]
    )

    review = critic_chain.invoke(
        {
            "question": state["question"],
            "evidence": state["formatted_evidence"],
            "answer": analysis.answer,
            "findings": format_findings(
                analysis.findings
            ),
        }
    )

    _stage_done(
        thread_id,
        "critic",
        "Research quality checked",
        f"Critic score: {review.score}/10",
    )

    return {
        "critic_review": review.model_dump(),
    }

def human_approval_node(state):

    thread_id = state["thread_id"]

    analysis = EvidenceAnalysis.model_validate(
        state["analysis"]
    )
    critic = CriticReview.model_validate(
        state["critic_review"]
    )

    _stage_waiting(
        thread_id,
        "human_review",
        "Waiting for your review",
        f"Critic score: {critic.score}/10 • "
        "Choose Approve, Research More, or Stop.",
    )

    decision = interrupt(
        {
            "question": state["question"],
            "answer": analysis.answer,
            "critic_score": critic.score,
            "critic_feedback": critic.feedback,
            "limitations": analysis.limitations,
            "message": (
                "Choose: approve, research_more, or stop."
            ),
        }
    )

    return {
        "human_decision": decision
    }

def route_after_human(state):

    decision = (
        state.get(
            "human_decision",
            ""
        )
        .lower()
        .strip()
    )

    if decision == "approve":
        return "approve"

    if decision == "research_more":
        return "research_more"

    return "stop"


# ============================================================
# 13. QUERY REFINEMENT
# ============================================================

refinement_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a research query refinement agent.

Use critic feedback to identify what information
is missing and create 2 to 4 precise searches.

Prefer authoritative technical documentation,
academic papers, and primary sources when relevant.
Do not repeat the original broad query.
""",
        ),
        (
            "human",
            """
QUESTION:
{question}

CRITIC FEEDBACK:
{feedback}

MISSING INFORMATION:
{missing_information}

Create improved search queries.
""",
        ),
    ]
)

refinement_chain = (
    refinement_prompt
    | get_llm().with_structured_output(
        QueryRefinement
    )
)


def refine_query_node(state):

    thread_id = state["thread_id"]

    _stage_start(
        thread_id,
        "refinement",
        "Refining research queries",
        "Using critic feedback to target missing information.",
    )

    critic = CriticReview.model_validate(
        state["critic_review"]
    )

    refinement = refinement_chain.invoke(
        {
            "question": state["question"],
            "feedback": critic.feedback,
            "missing_information": "\n".join(
                critic.missing_information
            ),
        }
    )

    _stage_done(
        thread_id,
        "refinement",
        "Research queries refined",
        f"{len(refinement.search_queries)} targeted queries created.",
    )

    return {
        "search_queries": refinement.search_queries,
        "refinement_reasoning": refinement.reasoning,
    }

def increment_retry_node(state):

    retry_count = state.get("retry_count", 0) + 1

    _set_progress(
        state["thread_id"],
        "refinement",
        "running",
        "Starting additional research",
        f"Re-research attempt {retry_count}.",
    )

    return {
        "retry_count": retry_count
    }

def research_more_node(state):

    thread_id = state["thread_id"]
    queries = state.get("search_queries", [])

    if not queries:
        _stage_done(
            thread_id,
            "web_research",
            "Additional research skipped",
            "No refined queries were produced.",
        )
        return {}

    _stage_start(
        thread_id,
        "web_research",
        "Running additional research",
        f"Searching {len(queries)} refined queries.",
    )

    results = deduplicate_results(
        search_queries(queries, max_results=5)
    )

    _set_progress(
        thread_id,
        "web_research",
        "running",
        "Extracting additional sources",
        f"{len(results)} unique results found.",
    )

    urls = [
        result["url"]
        for result in results[:5]
        if result.get("url")
    ]

    extracted = extract_web_content(urls)
    chunks = chunk_sources(extracted)

    # Fallback to search result snippets if web extraction returned no chunks
    if not chunks and results:
        chunks = [
            {
                "url": r.get("url", ""),
                "chunk_id": idx,
                "content": r.get("content", "").strip(),
            }
            for idx, r in enumerate(results)
            if r.get("content", "").strip()
        ]

    _set_progress(
        thread_id,
        "web_research",
        "running",
        "Updating evidence index",
        f"{len(chunks)} new chunks prepared.",
    )

    add_chunks_to_vectorstore(
        chunks,
        "retry_research",
        thread_id=thread_id,
    )

    docs = retrieve_evidence(state["question"], thread_id=thread_id)

    serializable_docs = [
        {
            "page_content": doc.page_content,
            "metadata": dict(doc.metadata),
        }
        for doc in docs
    ]

    _stage_done(
        thread_id,
        "web_research",
        "Additional research completed",
        f"{len(results)} results • {len(chunks)} chunks • "
        f"{len(docs)} evidence blocks",
    )

    return {
        "retrieved_docs": serializable_docs,
        "formatted_evidence": format_documents(docs),
        "source_count": len(results),
        "chunk_count": len(chunks),
        "retrieved_count": len(docs),
    }

def route_after_retry(state):

    # max_retries represents the maximum number of
    # additional research cycles allowed.
    if (
        state.get("retry_count", 0)
        > state.get(
            "max_retries",
            MAX_RETRIES,
        )
    ):
        return "stop"

    return "continue"


# ============================================================
# 15. REPORT GENERATION
# ============================================================

report_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are a professional research report writer.

Create a clear, professional report using ONLY
the provided research analysis.

Do not invent facts.
Do not add unsupported claims.
Clearly state limitations.
""",
        ),
        (
            "human",
            """
QUESTION:
{question}

ANSWER:
{answer}

FINDINGS:
{findings}

LIMITATIONS:
{limitations}

Create the final research report.
""",
        ),
    ]
)

report_chain = (
    report_prompt
    | get_llm().with_structured_output(
        ResearchReport
    )
)


def create_markdown(report, analysis, question):

    lines = [
        f"# {report.title}",
        "",
        "## Executive Summary",
        "",
        report.executive_summary,
        "",
        "## Research Question",
        "",
        question,
        "",
        "## Key Findings",
        "",
    ]

    lines.extend(
        f"{i}. {finding}"
        for i, finding in enumerate(
            report.key_findings,
            1,
        )
    )

    lines += [
        "",
        "## Detailed Analysis",
        "",
        report.detailed_analysis,
        "",
        "## Evidence Sources",
        "",
    ]

    lines.extend(
        f"{i}. {finding.supporting_source}"
        for i, finding in enumerate(
            analysis.findings,
            1,
        )
    )

    lines += [
        "",
        "## Limitations",
        "",
    ]

    lines.extend(
        f"- {item}"
        for item in report.limitations
    )

    lines += [
        "",
        "## Conclusion",
        "",
        report.conclusion,
        "",
        "---",
        "",
        "*Generated by Autonomous Research Agent*",
    ]

    return "\n".join(lines)


def create_pdf(report, analysis, question, path):

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ResearchTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=21,
        spaceAfter=20,
    )

    heading_style = ParagraphStyle(
        "ResearchHeading",
        parent=styles["Heading2"],
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = styles["BodyText"]

    def paragraph(text):
        return Paragraph(
            escape(str(text)).replace(
                "\n",
                "<br/>",
            ),
            body_style,
        )

    doc = SimpleDocTemplate(
        str(path),
        pagesize=A4,
        rightMargin=50,
        leftMargin=50,
        topMargin=50,
        bottomMargin=50,
    )

    story = [
        Paragraph(
            escape(report.title),
            title_style,
        ),
        Paragraph(
            "Research Question",
            heading_style,
        ),
        paragraph(question),
        Paragraph(
            "Executive Summary",
            heading_style,
        ),
        paragraph(
            report.executive_summary
        ),
        Paragraph(
            "Key Findings",
            heading_style,
        ),
    ]

    for i, finding in enumerate(
        report.key_findings,
        1,
    ):
        story.append(
            paragraph(
                f"{i}. {finding}"
            )
        )
        story.append(
            Spacer(
                1,
                0.08 * inch,
            )
        )

    story += [
        Paragraph(
            "Detailed Analysis",
            heading_style,
        ),
        paragraph(
            report.detailed_analysis
        ),
        Paragraph(
            "Evidence Sources",
            heading_style,
        ),
    ]

    for finding in analysis.findings:
        story.append(
            paragraph(
                finding.supporting_source
            )
        )

    story.append(
        Paragraph(
            "Limitations",
            heading_style,
        )
    )

    for item in report.limitations:
        story.append(
            paragraph(f"• {item}")
        )

    story += [
        Paragraph(
            "Conclusion",
            heading_style,
        ),
        paragraph(report.conclusion),
    ]

    doc.build(story)


def generate_report_node(state):

    thread_id = state["thread_id"]

    _stage_start(
        thread_id,
        "report",
        "Generating final report",
        "Writing the approved research into Markdown and PDF.",
    )

    analysis = EvidenceAnalysis.model_validate(
        state["analysis"]
    )

    report = report_chain.invoke(
        {
            "question": state["question"],
            "answer": analysis.answer,
            "findings": format_findings(
                analysis.findings
            ),
            "limitations": "\n".join(
                analysis.limitations
            ),
        }
    )

    markdown = create_markdown(
        report,
        analysis,
        state["question"],
    )

    safe_thread_id = "".join(
        ch if ch.isalnum() or ch in "-_" else "_"
        for ch in state.get("thread_id", "research")
    )

    markdown_path = REPORT_DIR / f"{safe_thread_id}.md"
    pdf_path = REPORT_DIR / f"{safe_thread_id}.pdf"

    markdown_path.write_text(
        markdown,
        encoding="utf-8",
    )

    create_pdf(
        report,
        analysis,
        state["question"],
        pdf_path,
    )

    _stage_done(
        thread_id,
        "report",
        "Research report generated",
        "Markdown and PDF reports are ready.",
    )

    return {
        "markdown_report": markdown,
        "markdown_path": str(markdown_path),
        "pdf_path": str(pdf_path),
    }

_compiled_graph = None
_graph_lock = threading.Lock()


def _build_graph():
    db_path = (
        CHECKPOINT_DIR
        / "research_agent.db"
    )

    connection = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(
        connection
    )
    checkpointer.setup()

    builder = StateGraph(
        ResearchState
    )

    builder.add_node(
        "planner",
        planner_node,
    )

    builder.add_node(
        "research",
        research_node,
    )

    builder.add_node(
        "analyst",
        analyst_node,
    )

    builder.add_node(
        "critic",
        critic_node,
    )

    builder.add_node(
        "human_approval",
        human_approval_node,
    )

    builder.add_node(
        "increment_retry",
        increment_retry_node,
    )

    builder.add_node(
        "refine_query",
        refine_query_node,
    )

    builder.add_node(
        "research_more",
        research_more_node,
    )

    builder.add_node(
        "generate_report",
        generate_report_node,
    )

    builder.add_edge(
        START,
        "planner",
    )

    builder.add_edge(
        "planner",
        "research",
    )

    builder.add_edge(
        "research",
        "analyst",
    )

    builder.add_edge(
        "analyst",
        "critic",
    )

    builder.add_edge(
        "critic",
        "human_approval",
    )

    builder.add_conditional_edges(
        "human_approval",
        route_after_human,
        {
            "approve":
                "generate_report",
            "research_more":
                "increment_retry",
            "stop":
                END,
        },
    )

    builder.add_edge(
        "increment_retry",
        "refine_query",
    )

    builder.add_edge(
        "refine_query",
        "research_more",
    )

    builder.add_conditional_edges(
        "research_more",
        route_after_retry,
        {
            "continue":
                "analyst",
            "stop":
                END,
        },
    )

    builder.add_edge(
        "generate_report",
        END,
    )

    return builder.compile(
        checkpointer=checkpointer
    )


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        with _graph_lock:
            if _compiled_graph is None:
                _compiled_graph = _build_graph()
    return _compiled_graph



# ============================================================
# 17. PUBLIC BACKEND API
# ============================================================

def start_research(
    question: str,
    thread_id: str,
    max_retries: int = MAX_RETRIES,
):

    _new_progress(thread_id)

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        return get_graph().invoke(
            {
                "question": question,
                "thread_id": thread_id,
                "retry_count": 0,
                "max_retries": max_retries,
            },
            config=config,
        )
    except Exception as exc:
        _set_progress(
            thread_id,
            "error",
            "failed",
            "Research failed",
            str(exc),
        )
        raise

def resume_research(
    thread_id: str,
    decision: Literal[
        "approve",
        "research_more",
        "stop",
    ],
):

    if decision == "approve":
        _set_progress(
            thread_id,
            "human_review",
            "done",
            "Research approved",
            "Continuing to final report generation.",
        )
    elif decision == "research_more":
        _set_progress(
            thread_id,
            "human_review",
            "done",
            "More research requested",
            "Using critic feedback to refine the search.",
        )
    else:
        _set_progress(
            thread_id,
            "human_review",
            "stopped",
            "Research stopped",
            "No final report was generated.",
        )

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        return get_graph().invoke(
            Command(resume=decision),
            config=config,
        )
    except Exception as exc:
        _set_progress(
            thread_id,
            "error",
            "failed",
            "Research continuation failed",
            str(exc),
        )
        raise

def get_current_state(
    thread_id: str,
):

    config = {
        "configurable": {
            "thread_id":
                thread_id
        }
    }

    return get_graph().get_state(
        config
    )


def needs_human_approval(
    thread_id: str,
):

    state = get_current_state(
        thread_id
    )

    return bool(
        state.next
        and
        "human_approval"
        in state.next
    )
