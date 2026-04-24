from agents import build_reader_agent, build_search_agent, classifier_chain, critic_chain, llm, writer_chain
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


direct_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's question clearly and concisely."),
        ("human", "{query}"),
    ]
)
direct_chain = direct_prompt | llm | StrOutputParser()


def _emit(progress_callback, step, status, extra=None):
    if progress_callback is not None:
        progress_callback(step, status, extra or {})


def _run_search(topic: str, instruction: str) -> str:
    search_agent = build_search_agent()
    result = search_agent.invoke({"messages": [("user", instruction.format(topic=topic))]})
    return result["messages"][-1].content


def _run_reader(topic: str, search_results: str) -> str:
    reader_agent = build_reader_agent()
    result = reader_agent.invoke(
        {
            "messages": [
                (
                    "user",
                    (
                        f"Based on the following search results about '{topic}', "
                        "pick the most relevant URL and scrape it for deeper content.\n\n"
                        f"Search Results:\n{search_results[:800]}"
                    ),
                )
            ]
        }
    )
    return result["messages"][-1].content


def _write_report(topic: str, research: str) -> str:
    return writer_chain.invoke({"topic": topic, "research": research})


def _critique_report(report: str) -> str:
    return critic_chain.invoke({"report": report})


def run_research_pipeline(topic: str, progress_callback=None) -> dict:
    state = {}

    query_type = classifier_chain.invoke({"query": topic}).strip().upper()
    if query_type not in {"SIMPLE", "FINANCE", "NEWS", "ACADEMIC", "RESEARCH"}:
        query_type = "RESEARCH"
    state["query_type"] = query_type
    _emit(progress_callback, "classify", "done", {"query_type": query_type})

    if query_type == "SIMPLE":
        print("\n[Router] Simple query detected. Skipping deep research.")
        state["direct_answer"] = direct_chain.invoke({"query": topic})
        return state

    if query_type == "FINANCE":
        print("\n[Router] Finance query detected.")
        _emit(progress_callback, "search", "running")
        state["search_results"] = _run_search(
            topic,
            (
                "Get current financial market data for: {topic}. "
                "Use the financial_data tool when possible, infer the most likely ticker if needed, "
                "and include recent context that helps explain the numbers."
            ),
        )
        _emit(progress_callback, "search", "done")

        _emit(progress_callback, "write", "running")
        state["report"] = _write_report(
            topic,
            f"FINANCIAL RESEARCH RESULTS:\n{state['search_results']}",
        )
        _emit(progress_callback, "write", "done")

        _emit(progress_callback, "critique", "running")
        state["feedback"] = _critique_report(state["report"])
        _emit(progress_callback, "critique", "done")
        return state

    if query_type == "NEWS":
        print("\n[Router] News query detected.")
        _emit(progress_callback, "search", "running")
        state["search_results"] = _run_search(
            topic,
            (
                "Find the latest news and breaking updates about: {topic}. "
                "Prefer the news_search tool and focus on the most recent developments."
            ),
        )
        _emit(progress_callback, "search", "done")

        _emit(progress_callback, "write", "running")
        state["report"] = _write_report(
            topic,
            f"NEWS RESULTS:\n{state['search_results']}",
        )
        _emit(progress_callback, "write", "done")

        _emit(progress_callback, "critique", "running")
        state["feedback"] = _critique_report(state["report"])
        _emit(progress_callback, "critique", "done")
        return state

    if query_type == "ACADEMIC":
        print("\n[Router] Academic query detected.")
        _emit(progress_callback, "search", "running")
        state["search_results"] = _run_search(
            topic,
            (
                "Find the most relevant academic and scientific material about: {topic}. "
                "Prefer arXiv for papers and Wikipedia for background context, and include useful URLs."
            ),
        )
        _emit(progress_callback, "search", "done")

        _emit(progress_callback, "scrape", "running")
        state["scraped_content"] = _run_reader(topic, state["search_results"])
        _emit(progress_callback, "scrape", "done")

        _emit(progress_callback, "write", "running")
        state["report"] = _write_report(
            topic,
            (
                f"ACADEMIC SEARCH RESULTS:\n{state['search_results']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
            ),
        )
        _emit(progress_callback, "write", "done")

        _emit(progress_callback, "critique", "running")
        state["feedback"] = _critique_report(state["report"])
        _emit(progress_callback, "critique", "done")
        return state

    print("\n[Router] Research query detected.")
    _emit(progress_callback, "search", "running")
    state["search_results"] = _run_search(
        topic,
        "Find recent, reliable and detailed information about: {topic}",
    )
    _emit(progress_callback, "search", "done")

    _emit(progress_callback, "scrape", "running")
    state["scraped_content"] = _run_reader(topic, state["search_results"])
    _emit(progress_callback, "scrape", "done")

    _emit(progress_callback, "write", "running")
    state["report"] = _write_report(
        topic,
        (
            f"SEARCH RESULTS:\n{state['search_results']}\n\n"
            f"DETAILED SCRAPED CONTENT:\n{state['scraped_content']}"
        ),
    )
    _emit(progress_callback, "write", "done")

    _emit(progress_callback, "critique", "running")
    state["feedback"] = _critique_report(state["report"])
    _emit(progress_callback, "critique", "done")
    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
