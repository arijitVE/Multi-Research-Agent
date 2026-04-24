from agents import chat_chain
from pipeline import run_research_pipeline
from services.session_service import create_session, get_chat_history, save_message, update_session


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
    return {"session_id": session.id, "state": _normalize_state(raw_state)}


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
