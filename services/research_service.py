from agents import chat_chain, critic_chain, improver_chain
from pipeline import run_research_pipeline
from services.session_service import (
    create_session,
    get_chat_history,
    get_session,
    increment_session_version,
    save_message,
    save_report_version,
    update_session,
)


def _normalize_state(raw_state):
    state = dict(raw_state)
    if "search_results" in raw_state:
        state["search"] = raw_state["search_results"]
    if "scraped_content" in raw_state:
        state["reader"] = raw_state["scraped_content"]
    if "report" in raw_state:
        state["writer"] = raw_state["report"]
    if "feedback" in raw_state:
        state["critic"] = raw_state["feedback"]
    return state


def run_research(topic: str, progress_callback=None) -> dict:
    raw_state = run_research_pipeline(topic, progress_callback=progress_callback)
    session = create_session(topic, raw_state.get("query_type"))
    update_session(
        session.id,
        report=raw_state.get("report") or raw_state.get("direct_answer"),
        feedback=raw_state.get("feedback"),
    )
    save_report_version(
        session_id=session.id,
        version_number=1,
        report=raw_state.get("report", ""),
        feedback=raw_state.get("feedback", ""),
        improvement_prompt=None,
        evaluator_score=raw_state.get("evaluator_score"),
        evaluator_passed=raw_state.get("evaluator_passed"),
    )
    normalized_state = _normalize_state(raw_state)
    normalized_state["current_version"] = 1
    return {
        "session_id": session.id,
        "state": normalized_state,
        "evaluator_score": raw_state.get("evaluator_score"),
        "evaluator_passed": raw_state.get("evaluator_passed"),
    }


def send_chat_message(session_id: str, question: str, report: str) -> str:
    history = get_chat_history(session_id)
    history_tuples = [
        ("human" if item["role"] == "user" else "assistant", item["content"])
        for item in history
    ]
    reply = chat_chain.invoke(
        {"report": report, "history": history_tuples, "question": question}
    )
    save_message(session_id, "user", question)
    save_message(session_id, "assistant", reply)
    return reply


def improve_report(session_id: str, instructions: str, use_critic: bool = False) -> dict:
    session = get_session(session_id)
    if not session:
        raise ValueError(f"Session {session_id} not found")

    current_report = session.report or ""
    current_feedback = session.feedback or ""

    if use_critic and current_feedback:
        full_instructions = (
            f"Apply the following critic feedback:\n{current_feedback}\n\n"
            f"Additional instructions:\n{instructions}"
            if instructions
            else f"Apply the following critic feedback:\n{current_feedback}"
        )
    else:
        full_instructions = instructions

    new_report = improver_chain.invoke(
        {
            "report": current_report,
            "feedback": current_feedback,
            "instructions": full_instructions,
        }
    )
    new_feedback = critic_chain.invoke({"report": new_report})

    updated_session = increment_session_version(
        session_id=session_id,
        new_report=new_report,
        new_feedback=new_feedback,
        improvement_prompt=full_instructions,
    )

    return {
        "report": new_report,
        "feedback": new_feedback,
        "version": updated_session.current_version,
    }
