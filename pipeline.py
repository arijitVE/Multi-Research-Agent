import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from agents import (
    classifier_chain,
    critic_chain,
    evaluator_chain,
    llm,
    query_decomposer_chain,
    writer_chain,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from tools import arxiv_search, financial_data, news_search, web_scrape, web_search, wikipedia_search


direct_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Answer the user's question clearly and concisely."),
        ("human", "{query}"),
    ]
)
direct_chain = direct_prompt | llm | StrOutputParser()


def safe_json_parse(text: str) -> dict:
    try:
        return json.loads(text.strip())
    except Exception:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                return {}
    return {}


def run_concurrent_search(queries: list[str], query_type: str) -> str:
    """Run multiple searches concurrently and combine results."""
    tool_map = {
        "FINANCE": [financial_data, web_search, news_search],
        "NEWS": [news_search, web_search],
        "ACADEMIC": [arxiv_search, wikipedia_search, web_search],
        "RESEARCH": [web_search, news_search, wikipedia_search],
    }
    tasks = []
    if query_type == "FINANCE":
        tasks = [
            (financial_data, queries[0]),
            (web_search, queries[0]),
            (news_search, queries[2]),
        ]
    else:
        selected_tools = tool_map.get(query_type, [web_search])
        for tool_fn in selected_tools:
            if tool_fn is news_search:
                relevant_queries = [queries[2]]
                if query_type == "NEWS":
                    relevant_queries.insert(0, queries[0])
            elif tool_fn is web_search:
                relevant_queries = queries[:2]
            elif tool_fn is arxiv_search:
                relevant_queries = queries[:2]
            elif tool_fn is wikipedia_search:
                relevant_queries = queries[:2]
            else:
                relevant_queries = queries
            for query in relevant_queries:
                tasks.append((tool_fn, query))
        tasks = tasks[:6]

    results = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_map = {
            executor.submit(tool_fn.invoke, query_arg): (tool_fn.name, query_arg)
            for tool_fn, query_arg in tasks
        }
        for future in as_completed(future_map):
            tool_name, query_arg = future_map[future]
            try:
                result = future.result()
            except Exception as e:
                result = f"{tool_name} failed for {query_arg}: {e}"
            results.append(f"=== {tool_name.upper()} RESULTS ===\n{result}\n")

    print(f"\n[Search] Gathered {len(results)} source batches.")
    return "\n".join(results)


def run_concurrent_scrape(search_results: str) -> str:
    """Extract top 3 URLs from search results and scrape concurrently."""
    urls = []
    for url in re.findall(r"URL:\s*(https?://\S+)", search_results):
        cleaned = url.strip().rstrip(").,]")
        if cleaned not in urls and "arxiv.org/abs/" not in cleaned:
            urls.append(cleaned)
        if len(urls) == 3:
            break

    if not urls:
        print("\n[Scrape] No URLs found to scrape.")
        return ""

    sections = []
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_map = {executor.submit(web_scrape.invoke, url): url for url in urls}
        try:
            for future in as_completed(future_map, timeout=20):
                url = future_map[future]
                try:
                    content = future.result(timeout=15)
                except Exception as e:
                    content = f"Scrape failed for {url}: {e}"
                sections.append(f"=== SCRAPED: {url[:60]} ===\n{content}\n")
        except Exception as e:
            sections.append(f"=== SCRAPE ERROR ===\n{e}\n")

    print(f"\n[Scrape] Scraped {len(urls)} URLs.")
    return "\n".join(sections)


def evaluate_data(topic: str, data: str) -> dict:
    """Run internal evaluator. Returns parsed dict with passed, score, retry_query."""
    try:
        raw = evaluator_chain.invoke({"topic": topic, "data": data[:6000]})
        parsed = safe_json_parse(raw)
        if not parsed or parsed.get("score", 50) < 10:
            parsed = {"passed": True, "score": 50}
    except Exception:
        parsed = {"passed": True, "score": 50}

    print(
        f"\n[Evaluator] score={parsed.get('score', 50)} "
        f"passed={parsed.get('passed', True)}"
    )
    return parsed


def run_research_pipeline(topic: str, progress_callback=None) -> dict:
    def emit(step, status, extra=None):
        if progress_callback:
            progress_callback(step, status, extra)

    state = {}

    emit("classify", "running")
    query_type = classifier_chain.invoke({"query": topic}).strip().upper()
    if query_type not in ("SIMPLE", "FINANCE", "NEWS", "ACADEMIC", "RESEARCH"):
        query_type = "RESEARCH"
    state["query_type"] = query_type
    emit("classify", "done", {"query_type": query_type})

    if query_type == "SIMPLE":
        emit("answer", "running")
        state["direct_answer"] = direct_chain.invoke({"query": topic})
        emit("answer", "done")
        return state

    emit("decompose", "running")
    try:
        decomposed_raw = query_decomposer_chain.invoke({"topic": topic})
        decomposed = safe_json_parse(decomposed_raw)
        queries = [
            decomposed.get("primary", topic),
            decomposed.get("secondary", topic + " analysis"),
            decomposed.get("tertiary", topic + " latest 2025"),
        ]
    except Exception:
        queries = [topic, topic + " analysis", topic + " latest news 2025"]
    emit("decompose", "done", {"queries": queries})
    print(f"\n[Decomposer] 3 search angles: {queries}")

    emit("search", "running")
    max_attempts = 2
    gathered_data = ""
    search_data = ""
    scraped_data = ""
    evaluator_result = {"passed": True, "score": 100}

    for attempt in range(max_attempts):
        if attempt == 1:
            retry_q = evaluator_result.get("retry_query", "")
            if retry_q:
                queries[0] = retry_q
            print(f"\n[Evaluator] Retry attempt with refined query: {queries[0]}")
            emit("retry", "running", {"reason": evaluator_result.get("missing", "")})

        search_data = run_concurrent_search(queries, query_type)

        scraped_data = ""
        if query_type not in ("FINANCE", "NEWS"):
            emit("scrape", "running")
            scraped_data = run_concurrent_scrape(search_data)
            emit("scrape", "done")

        gathered_data = search_data + "\n\n" + scraped_data

        emit("evaluate", "running")
        evaluator_result = evaluate_data(topic, gathered_data)
        state["evaluator_score"] = evaluator_result.get("score", 0)
        state["evaluator_passed"] = evaluator_result.get("passed", True)
        emit(
            "evaluate",
            "done",
            {
                "score": state["evaluator_score"],
                "passed": state["evaluator_passed"],
            },
        )

        if evaluator_result.get("passed", True):
            print(f"\n[Evaluator] Passed with score {state['evaluator_score']}/100")
            break

        print(
            f"\n[Evaluator] Failed (score {state['evaluator_score']}/100). "
            f"Missing: {evaluator_result.get('missing', '')}"
        )
        if attempt == max_attempts - 1:
            print("[Evaluator] Max retries reached. Proceeding with available data.")

    emit("search", "done")
    state["search_results"] = search_data
    state["scraped_content"] = scraped_data

    emit("write", "running")
    state["report"] = writer_chain.invoke(
        {
            "topic": topic,
            "research": gathered_data[:6000],
        }
    )
    emit("write", "done")
    print("\n[Writer] Report drafted.")

    emit("critique", "running")
    state["feedback"] = critic_chain.invoke({"report": state["report"]})
    emit("critique", "done")
    print("\n[Critic] Feedback generated.")

    return state


if __name__ == "__main__":
    topic = input("\n Enter a research topic : ")
    run_research_pipeline(topic)
