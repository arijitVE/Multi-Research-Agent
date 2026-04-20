import streamlit as st
import time
from agents import build_reader_agent, build_search_agent, writer_chain, critic_chain, classifier_chain, llm
from pipeline import direct_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}
.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

.hero { text-align: center; padding: 3.5rem 0 2.5rem; position: relative; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.25em; text-transform: uppercase; color: #ff8c32;
    margin-bottom: 1rem; opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif; font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800; line-height: 1.0; letter-spacing: -0.03em;
    color: #f0ebe0; margin: 0 0 1rem;
}
.hero h1 span { color: #ff8c32; }
.hero-sub {
    font-size: 1.05rem; font-weight: 300; color: #a09890;
    max-width: 520px; margin: 0 auto; line-height: 1.65;
}
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2rem 0;
}
.input-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,140,50,0.15);
    border-radius: 16px; padding: 2rem 2.5rem; margin-bottom: 2rem;
    backdrop-filter: blur(8px);
}
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 10px !important; color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255,140,50,0.12) !important;
}
.stTextInput > label {
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    color: #ff8c32 !important; font-weight: 500 !important;
}
.stTextArea > label {
    font-family: 'DM Mono', monospace !important; font-size: 0.72rem !important;
    letter-spacing: 0.15em !important; text-transform: uppercase !important;
    color: #ff8c32 !important; font-weight: 500 !important;
}
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 10px !important; color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important;
    color: #0a0a0f !important; font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.95rem !important;
    letter-spacing: 0.04em !important; border: none !important;
    border-radius: 10px !important; padding: 0.7rem 2.2rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
    box-shadow: 0 4px 20px rgba(255,140,50,0.3) !important; width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,140,50,0.4) !important; opacity: 0.95 !important;
}
.stButton > button:active { transform: translateY(0) !important; }

.step-card {
    background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.5rem 1.8rem; margin-bottom: 1.2rem;
    position: relative; overflow: hidden; transition: border-color 0.3s;
}
.step-card.active { border-color: rgba(255,140,50,0.4); background: rgba(255,140,50,0.04); }
.step-card.done   { border-color: rgba(80,200,120,0.3);  background: rgba(80,200,120,0.03); }
.step-card::before {
    content: ''; position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    border-radius: 14px 0 0 14px; background: rgba(255,255,255,0.05); transition: background 0.3s;
}
.step-card.active::before { background: #ff8c32; }
.step-card.done::before   { background: #50c878; }
.step-header { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.3rem; }
.step-num  { font-family: 'DM Mono', monospace; font-size: 0.68rem; font-weight: 500; letter-spacing: 0.15em; color: #ff8c32; opacity: 0.7; }
.step-title { font-family: 'Syne', sans-serif; font-size: 0.95rem; font-weight: 700; color: #f0ebe0; }
.step-status { margin-left: auto; font-family: 'DM Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em; }
.status-waiting { color: #555; }
.status-running { color: #ff8c32; }
.status-done    { color: #50c878; }

.result-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px; padding: 1.8rem 2rem; margin-top: 1rem; margin-bottom: 1.5rem;
}
.result-panel-title {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; font-weight: 500;
    letter-spacing: 0.2em; text-transform: uppercase; color: #ff8c32;
    margin-bottom: 1rem; padding-bottom: 0.7rem;
    border-bottom: 1px solid rgba(255,140,50,0.15);
}
.result-content {
    font-size: 0.92rem; line-height: 1.8; color: #cdc8bf;
    white-space: pre-wrap; font-family: 'DM Sans', sans-serif;
}
.report-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(255,140,50,0.2);
    border-radius: 16px; padding: 2rem 2.5rem; margin-top: 1rem;
}
.feedback-panel {
    background: rgba(255,255,255,0.025); border: 1px solid rgba(80,200,120,0.2);
    border-radius: 16px; padding: 2rem 2.5rem; margin-top: 1rem;
}
.panel-label {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em;
    text-transform: uppercase; margin-bottom: 1.2rem; padding-bottom: 0.7rem;
}
.panel-label.orange { color: #ff8c32; border-bottom: 1px solid rgba(255,140,50,0.15); }
.panel-label.green  { color: #50c878; border-bottom: 1px solid rgba(80,200,120,0.15); }

/* Action bar */
.action-bar {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,140,50,0.12);
    border-radius: 14px; padding: 1.5rem 2rem; margin-top: 1.5rem;
}
.action-bar-title {
    font-family: 'DM Mono', monospace; font-size: 0.7rem; letter-spacing: 0.2em;
    text-transform: uppercase; color: #ff8c32; margin-bottom: 1rem;
}
.custom-prompt-box {
    background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px; padding: 1.2rem 1.5rem; margin-top: 1rem;
}

.stSpinner > div { color: #ff8c32 !important; }
details summary {
    font-family: 'DM Mono', monospace !important; font-size: 0.75rem !important;
    color: #a09890 !important; letter-spacing: 0.1em !important; cursor: pointer;
}
.section-heading {
    font-family: 'Syne', sans-serif; font-size: 1.3rem; font-weight: 700;
    color: #f0ebe0; margin: 2rem 0 1rem;
}
.notice {
    font-family: 'DM Mono', monospace; font-size: 0.72rem; color: #605850;
    text-align: center; margin-top: 3rem; letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)


# ── Helper: step card ─────────────────────────────────────────────────────────
def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING",   "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done":    ("✓ DONE",    "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        {"<div style='font-size:0.82rem;color:#706860;margin-top:0.3rem;'>"+desc+"</div>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)


# ── Helper: run a quick action chain ─────────────────────────────────────────
def run_action(system_prompt: str, content: str) -> str:
    p = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{content}")
    ])
    return (p | llm | StrOutputParser()).invoke({"content": content})


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("results", {}),
    ("running", False),
    ("done", False),
    ("action_result", None),
    ("action_label", ""),
    ("show_followup", False),
    ("show_custom", False),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">
        Four specialized AI agents collaborate — searching, scraping, writing,
        and critiquing — to deliver a polished research report on any topic.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout: input left, pipeline right ───────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. Quantum computing breakthroughs in 2025",
        key="topic_input",
        label_visibility="visible",
    )
    run_btn = st.button("⚡  Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem;">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#605850;letter-spacing:0.1em;">TRY →</span>
    """, unsafe_allow_html=True)
    for ex in ["LLM agents 2025", "CRISPR gene editing", "Fusion energy progress"]:
        st.markdown(f"""
        <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:6px;padding:0.25rem 0.7rem;font-size:0.75rem;color:#a09890;
            font-family:'DM Sans',sans-serif;cursor:default;">{ex}</span>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    r = st.session_state.results

    def s(step):
        if not r:
            return "waiting"
        if step in r:
            return "done"
        if st.session_state.running:
            for k in ["search", "reader", "writer", "critic"]:
                if k not in r:
                    return "running" if k == step else "waiting"
        return "waiting"

    step_card("01", "Search Agent",  s("search"), "Gathers recent web information")
    step_card("02", "Reader Agent",  s("reader"), "Scrapes & extracts deep content")
    step_card("03", "Writer Chain",  s("writer"), "Drafts the full research report")
    step_card("04", "Critic Chain",  s("critic"), "Reviews & scores the report")


# ── Trigger pipeline ──────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.results      = {}
        st.session_state.running      = True
        st.session_state.done         = False
        st.session_state.action_result = None
        st.session_state.action_label  = ""
        st.session_state.show_followup = False
        st.session_state.show_custom   = False
        st.rerun()


# ── Execute pipeline ──────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.done:
    results   = {}
    topic_val = st.session_state.topic_input

    # Classify
    with st.spinner("🤔 Analysing your query…"):
        query_type = classifier_chain.invoke({"query": topic_val}).strip().upper()
        results["query_type"] = query_type

    if query_type == "SIMPLE":
        with st.spinner("💬 Answering directly…"):
            results["direct_answer"] = direct_chain.invoke({"query": topic_val})
        st.session_state.results = dict(results)
        st.session_state.running = False
        st.session_state.done    = True
        st.rerun()
    else:
        # Step 1 — Search
        with st.spinner("🔍 Search Agent is working…"):
            search_agent = build_search_agent()
            sr = search_agent.invoke({
                "messages": [("user", f"Find recent, reliable and detailed information about: {topic_val}")]
            })
            results["search"] = sr["messages"][-1].content
            st.session_state.results = dict(results)

        # Step 2 — Reader
        with st.spinner("📄 Reader Agent is scraping top resources…"):
            reader_agent = build_reader_agent()
            rr = reader_agent.invoke({
                "messages": [("user",
                    f"Based on the following search results about '{topic_val}', "
                    f"pick the most relevant URL and scrape it for deeper content.\n\n"
                    f"Search Results:\n{results['search'][:800]}"
                )]
            })
            results["reader"] = rr["messages"][-1].content
            st.session_state.results = dict(results)

        # Step 3 — Writer
        with st.spinner("✍️ Writer is drafting the report…"):
            research_combined = (
                f"SEARCH RESULTS:\n{results['search']}\n\n"
                f"DETAILED SCRAPED CONTENT:\n{results['reader']}"
            )
            results["writer"] = writer_chain.invoke({
                "topic":    topic_val,
                "research": research_combined,
            })
            st.session_state.results = dict(results)

        # Step 4 — Critic
        with st.spinner("🧐 Critic is reviewing the report…"):
            results["critic"] = critic_chain.invoke({"report": results["writer"]})
            st.session_state.results = dict(results)

        st.session_state.running = False
        st.session_state.done    = True
        st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── SIMPLE path ──
    if r.get("query_type") == "SIMPLE":
        st.markdown('<div class="section-heading">💬 Answer</div>', unsafe_allow_html=True)
        st.markdown(r.get("direct_answer", ""))

    # ── RESEARCH path ──
    else:
        st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)

        if "search" in r:
            with st.expander("🔍 Search Results (raw)", expanded=False):
                st.markdown(
                    f'<div class="result-panel">'
                    f'<div class="result-panel-title">Search Agent Output</div>'
                    f'<div class="result-content">{r["search"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if "reader" in r:
            with st.expander("📄 Scraped Content (raw)", expanded=False):
                st.markdown(
                    f'<div class="result-panel">'
                    f'<div class="result-panel-title">Reader Agent Output</div>'
                    f'<div class="result-content">{r["reader"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if "writer" in r:
            st.markdown('<div class="report-panel"><div class="panel-label orange">📝 Final Research Report</div>', unsafe_allow_html=True)
            st.markdown(r["writer"])
            st.markdown('</div>', unsafe_allow_html=True)

            st.download_button(
                label="⬇  Download Report (.md)",
                data=r["writer"],
                file_name=f"research_report_{int(time.time())}.md",
                mime="text/markdown",
            )

        if "critic" in r:
            st.markdown('<div class="feedback-panel"><div class="panel-label green">🧐 Critic Feedback</div>', unsafe_allow_html=True)
            st.markdown(r["critic"])
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Action bar (shown after any result) ──────────────────────────────────
    base_content = r.get("writer") or r.get("direct_answer", "")

    if base_content:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-bar"><div class="action-bar-title">⚡ What do you want to do next?</div>', unsafe_allow_html=True)

        # Fixed action buttons — row 1
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧒 Simplify for a 10-year-old", use_container_width=True):
                with st.spinner("Simplifying…"):
                    st.session_state.action_result = run_action(
                        "Rewrite this content so a 10-year-old can understand it. Use simple words and fun analogies.",
                        base_content,
                    )
                    st.session_state.action_label = "🧒 Simplified Version"
                    st.session_state.show_followup = False
                    st.session_state.show_custom   = False

        with col2:
            if st.button("🐦 Make a Twitter/X thread", use_container_width=True):
                with st.spinner("Threading…"):
                    st.session_state.action_result = run_action(
                        "Convert this into a compelling Twitter/X thread. Number each tweet. Max 280 chars each. Start with a strong hook.",
                        base_content,
                    )
                    st.session_state.action_label = "🐦 Twitter/X Thread"
                    st.session_state.show_followup = False
                    st.session_state.show_custom   = False

        with col3:
            if st.button("📊 Extract key bullet points", use_container_width=True):
                with st.spinner("Extracting…"):
                    st.session_state.action_result = run_action(
                        "Extract the 5–7 most important insights as concise, numbered bullet points.",
                        base_content,
                    )
                    st.session_state.action_label = "📊 Key Bullet Points"
                    st.session_state.show_followup = False
                    st.session_state.show_custom   = False

        # Fixed action buttons — row 2
        col4, col5 = st.columns(2)
        with col4:
            if st.button("❓ Ask a follow-up question", use_container_width=True):
                st.session_state.show_followup = not st.session_state.show_followup
                st.session_state.show_custom   = False

        with col5:
            if st.button("✏️ Custom prompt", use_container_width=True):
                st.session_state.show_custom   = not st.session_state.show_custom
                st.session_state.show_followup = False

        st.markdown('</div>', unsafe_allow_html=True)  # close action-bar

        # ── Follow-up question input ──────────────────────────────────────────
        if st.session_state.show_followup:
            st.markdown('<div class="custom-prompt-box">', unsafe_allow_html=True)
            followup = st.text_input("Your follow-up question:", key="followup_input", label_visibility="visible")
            if st.button("Submit question", key="submit_followup") and followup.strip():
                with st.spinner("Answering…"):
                    p = ChatPromptTemplate.from_messages([
                        ("system", "Answer the follow-up question based solely on the research context provided. Be specific and concise."),
                        ("human", "Context:\n{context}\n\nQuestion: {question}")
                    ])
                    st.session_state.action_result = (p | llm | StrOutputParser()).invoke({
                        "context":  base_content,
                        "question": followup,
                    })
                    st.session_state.action_label  = f"❓ Follow-up: {followup}"
                    st.session_state.show_followup = False
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Custom prompt input ───────────────────────────────────────────────
        if st.session_state.show_custom:
            st.markdown('<div class="custom-prompt-box">', unsafe_allow_html=True)
            st.markdown(
                "<p style='font-size:0.82rem;color:#a09890;margin-bottom:0.5rem;'>"
                "Write any instruction — e.g. <i>\"Translate to Hindi\"</i>, "
                "<i>\"Write an email pitch based on this\"</i>, "
                "<i>\"Find weaknesses in this argument\"</i>"
                "</p>",
                unsafe_allow_html=True,
            )
            custom_prompt = st.text_area(
                "Your custom instruction:",
                key="custom_prompt_input",
                height=100,
                placeholder="e.g. Rewrite this as a formal executive summary in under 200 words",
                label_visibility="visible",
            )
            if st.button("▶  Run custom prompt", key="submit_custom") and custom_prompt.strip():
                with st.spinner("Running your prompt…"):
                    p = ChatPromptTemplate.from_messages([
                        ("system", "{instruction}\n\nApply this to the content the user provides."),
                        ("human", "Content:\n{content}")
                    ])
                    st.session_state.action_result = (p | llm | StrOutputParser()).invoke({
                        "instruction": custom_prompt,
                        "content":     base_content,
                    })
                    st.session_state.action_label = f"✏️ Custom: {custom_prompt[:60]}…"
                    st.session_state.show_custom  = False
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Action result display ─────────────────────────────────────────────
        if st.session_state.get("action_result"):
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="result-panel">'
                f'<div class="result-panel-title">{st.session_state.action_label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.markdown(st.session_state.action_result)

            if st.button("🗑 Clear result", key="clear_action"):
                st.session_state.action_result = None
                st.session_state.action_label  = ""
                st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · Powered by LangChain multi-agent pipeline · Built with Streamlit
</div>
""", unsafe_allow_html=True)