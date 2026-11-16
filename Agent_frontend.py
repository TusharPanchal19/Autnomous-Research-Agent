
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import streamlit as st

from Agent_backend import (
    start_research,
    resume_research,
    get_current_state,
    get_progress,
    needs_human_approval,
)


st.set_page_config(
    page_title="Autonomous Research Agent",
    page_icon="🔬",
    layout="wide",
)


# ============================================================
# Styling
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
    }

    .stage-card {
        border: 1px solid rgba(128,128,128,.28);
        border-radius: 12px;
        padding: 14px 18px;
        margin: 8px 0;
    }

    .stage-title {
        font-size: 1.02rem;
        font-weight: 650;
    }

    .stage-detail {
        color: rgba(128,128,128,.95);
        margin-top: 5px;
        font-size: .90rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Session state
# ============================================================

defaults = {
    "thread_id": None,
    "result": None,
    "question": "",
    "max_retries": 2,
    "job_running": False,
    "job_error": None,
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

if "executor" not in st.session_state:
    # One local worker avoids multiple Ollama/embedding jobs
    # fighting for the laptop's limited VRAM.
    st.session_state.executor = ThreadPoolExecutor(
        max_workers=1
    )


# ============================================================
# Helpers
# ============================================================

def field(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


STAGE_NAMES = {
    "starting": "Starting",
    "planning": "Planning",
    "web_research": "Web Research",
    "analysis": "Evidence Analysis",
    "critic": "Research Critic",
    "refinement": "Query Refinement",
    "human_review": "Human Review",
    "report": "Report Generation",
    "error": "Research",
}


def render_stage_card(title, status, detail=""):
    icons = {
        "done": "✅",
        "running": "🔄",
        "waiting": "⏳",
        "failed": "❌",
        "stopped": "🛑",
    }

    icon = icons.get(status, "○")

    st.markdown(
        f"""
        <div class="stage-card">
            <div class="stage-title">{icon} {title}</div>
            <div class="stage-detail">{detail}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_live_progress(thread_id):
    progress = get_progress(thread_id)

    current_stage = progress.get("stage", "starting")
    current_status = progress.get("status", "running")
    current_message = progress.get(
        "message",
        "Starting research...",
    )
    current_detail = progress.get("detail", "")

    history = progress.get("history", [])

    st.subheader("Research Progress")

    # Completed work appears as a timeline.
    completed = set()

    for event in history:
        stage = event.get("stage")

        if stage in completed:
            continue

        completed.add(stage)

        detail = event.get("message", "")

        if event.get("detail"):
            detail += f" • {event['detail']}"

        render_stage_card(
            STAGE_NAMES.get(
                stage,
                stage.replace("_", " ").title(),
            ),
            event.get("status", "done"),
            detail,
        )

    # Only ONE active task is shown.
    if current_status == "running":
        title = STAGE_NAMES.get(
            current_stage,
            current_stage.replace("_", " ").title(),
        )

        with st.container(border=True):
            st.markdown(f"### 🔄 {title}")
            st.write(current_message)

            if current_detail:
                st.caption(current_detail)

            st.progress(50)

    elif current_status in {
        "waiting",
        "failed",
        "stopped",
    }:
        title = STAGE_NAMES.get(
            current_stage,
            current_stage.replace("_", " ").title(),
        )

        detail = current_message

        if current_detail:
            detail += f" • {current_detail}"

        render_stage_card(
            title,
            current_status,
            detail,
        )


def get_values(thread_id):
    try:
        snapshot = get_current_state(thread_id)
        return snapshot.values or {}
    except Exception:
        return {}


def render_report(result):
    markdown_path = result.get("markdown_path")
    pdf_path = result.get("pdf_path")
    markdown_text = result.get("markdown_report")

    if markdown_path:
        path = Path(markdown_path)

        if path.exists():
            try:
                markdown_text = path.read_text(
                    encoding="utf-8"
                )
            except Exception:
                pass

    if not markdown_text and not pdf_path:
        return

    st.divider()
    st.subheader("📄 Research Report")

    if markdown_text:
        with st.container(border=True):
            st.markdown(markdown_text)

    c1, c2 = st.columns(2)

    if markdown_path:
        path = Path(markdown_path)

        if path.exists():
            with c1:
                st.download_button(
                    "⬇️ Download Markdown",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="text/markdown",
                    use_container_width=True,
                )

    if pdf_path:
        path = Path(pdf_path)

        if path.exists():
            with c2:
                st.download_button(
                    "⬇️ Download PDF",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf",
                    use_container_width=True,
                )


def run_job_with_live_ui(
    thread_id,
    question,
    max_retries,
):
    future = st.session_state.executor.submit(
        start_research,
        question,
        thread_id,
        max_retries,
    )

    placeholder = st.empty()

    while not future.done():
        with placeholder.container():
            render_live_progress(thread_id)

        time.sleep(0.7)

    with placeholder.container():
        render_live_progress(thread_id)

    return future.result()


# ============================================================
# UI
# ============================================================

st.title("🔬 Autonomous Research Agent")
st.caption(
    "Plan → Research → Analyze → Critic → Human Review → Report"
)

st.subheader("Research Question")

question = st.text_area(
    "Research question",
    value=st.session_state.question,
    height=120,
    label_visibility="collapsed",
    placeholder=(
        "Example: Should a startup use a local LLM or a cloud LLM "
        "for a customer-support AI system?"
    ),
)

st.session_state.question = question

with st.expander("Research Settings"):
    st.session_state.max_retries = st.slider(
        "Maximum re-research attempts",
        min_value=0,
        max_value=3,
        value=st.session_state.max_retries,
    )


if st.button(
    "🚀 Start Research",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.job_running,
):
    if not question.strip():
        st.warning("Please enter a research question.")
    else:
        thread_id = "research_" + uuid.uuid4().hex

        st.session_state.thread_id = thread_id
        st.session_state.result = None
        st.session_state.job_error = None
        st.session_state.job_running = True

        try:
            result = run_job_with_live_ui(
                thread_id,
                question.strip(),
                st.session_state.max_retries,
            )

            st.session_state.result = result

        except Exception as exc:
            st.session_state.job_error = str(exc)

        finally:
            st.session_state.job_running = False

        st.rerun()


if st.session_state.job_error:
    st.error(
        "The research workflow failed. "
        "Check the terminal for the full traceback."
    )

    with st.expander("Error details"):
        st.code(st.session_state.job_error)


# ============================================================
# Progress & Human Review
# ============================================================

thread_id = st.session_state.thread_id
waiting = False

if thread_id and not st.session_state.job_running:
    # Preserve research progress timeline on screen
    render_live_progress(thread_id)

    try:
        waiting = needs_human_approval(thread_id)
    except Exception:
        waiting = False

    if waiting:
        values = get_values(thread_id)

        st.divider()
        st.subheader("👤 Human Review")

        analysis = values.get("analysis")
        critic = values.get("critic_review")

        if analysis:
            st.markdown("### Current Answer")
            st.write(field(analysis, "answer", ""))

        if critic:
            c1, c2 = st.columns(2)

            with c1:
                score = field(critic, "score")
                st.metric(
                    "Critic Score",
                    f"{score}/10" if score is not None else "—",
                )

            with c2:
                sufficient = field(
                    analysis,
                    "evidence_sufficient",
                    False,
                )
                st.metric(
                    "Evidence Sufficient",
                    "Yes" if sufficient else "No",
                )

            feedback = field(
                critic,
                "feedback",
                "",
            )

            if feedback:
                with st.expander(
                    "View Critic Feedback",
                    expanded=True,
                ):
                    st.write(feedback)

        c1, c2, c3 = st.columns(3)

        with c1:
            approve = st.button(
                "✅ Approve & Generate",
                use_container_width=True,
            )

        with c2:
            research_more = st.button(
                "🔄 Research More",
                use_container_width=True,
            )

        with c3:
            stop = st.button(
                "🛑 Stop",
                use_container_width=True,
            )

        decision = None

        if approve:
            decision = "approve"
        elif research_more:
            decision = "research_more"
        elif stop:
            decision = "stop"

        if decision:
            st.session_state.job_running = True

            try:
                future = st.session_state.executor.submit(
                    resume_research,
                    thread_id,
                    decision,
                )

                placeholder = st.empty()

                while not future.done():
                    with placeholder.container():
                        render_live_progress(thread_id)

                    time.sleep(0.7)

                with placeholder.container():
                    render_live_progress(thread_id)

                st.session_state.result = future.result()

            except Exception as exc:
                st.session_state.job_error = str(exc)

            finally:
                st.session_state.job_running = False

            st.rerun()


# ============================================================
# Final Results
# ============================================================

result = st.session_state.result

if result and not st.session_state.job_running and not waiting:
    st.divider()
    st.subheader("📊 Research Results")

    analysis = result.get("analysis")
    critic = result.get("critic_review")

    if analysis:
        st.markdown("### 📝 Answer")
        st.write(field(analysis, "answer", ""))

        findings = field(
            analysis,
            "findings",
            [],
        )

        if findings:
            st.markdown("### 🔎 Key Findings")

            for finding in findings:
                claim = field(
                    finding,
                    "claim",
                    str(finding),
                )

                source = field(
                    finding,
                    "supporting_source",
                    "",
                )

                st.markdown(f"- **{claim}**")

                if source:
                    st.caption(
                        f"Source: {source}"
                    )

        limitations = field(
            analysis,
            "limitations",
            [],
        )

        if limitations:
            with st.expander("⚠️ Limitations"):
                for item in limitations:
                    st.write(f"- {item}")

    if critic:
        st.markdown("### 🔍 Critic Review")

        score = field(critic, "score")
        approved = field(critic, "approved")
        feedback = field(critic, "feedback", "")

        c1, c2 = st.columns(2)

        with c1:
            st.metric(
                "Quality Score",
                f"{score}/10" if score is not None else "—",
            )

        with c2:
            st.metric(
                "Review Status",
                "Approved" if approved else "Needs Improvement",
            )

        if feedback:
            st.write(feedback)

    render_report(result)
