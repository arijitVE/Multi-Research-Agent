import streamlit as st
import time
from agents import llm
from db.database import init_db
from services.research_service import run_research, send_chat_message
from services.session_service import clear_chat_history, delete_session, get_all_sessions, get_session
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Persistent storage helpers ────────────────────────────────────────────────
init_db()


def build_results_from_session(session):
    results = {"query_type": session.query_type}
    if session.query_type == "SIMPLE":
        results["direct_answer"] = session.report or ""
    else:
        if session.report:
            results["writer"] = session.report
        if session.feedback:
            results["critic"] = session.feedback
    return results


def restore_session(session):
    topic = session.topic or ""
    chat_history = [{"role": m.role, "content": m.content} for m in session.messages]
    st.session_state.update({
        "results":           build_results_from_session(session),
        "chat_history":      chat_history,
        "active_session_id": session.id,
        "done":              True,
        "running":           False,
        "action_result":     None,
        "action_label":      "",
        "action_running":    False,
        "chat_open":         bool(chat_history),
        "topic_input":       topic,
        "_current_topic":    topic,
        "_pending_action":   "",
        "_pending_chat_q":   "",
    })

def fmt_date(ts):
    return ts.strftime("%b %d, %Y")


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; color:#e8e4dc; }
.stApp {
    background:#0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%,rgba(255,140,50,0.12) 0%,transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%,rgba(255,80,30,0.08) 0%,transparent 55%);
}
#MainMenu,footer { visibility:hidden; }
[data-testid="stHeader"] {
    background:transparent !important;
}
[data-testid="stToolbar"] {
    right:0.75rem !important;
}
[data-testid="collapsedControl"] {
    position:fixed !important;
    top:0.9rem !important;
    left:0.9rem !important;
    z-index:1000 !important;
}
[data-testid="collapsedControl"] button {
    background:rgba(13,13,20,0.92) !important;
    border:1px solid rgba(255,140,50,0.28) !important;
    border-radius:12px !important;
    box-shadow:0 10px 24px rgba(0,0,0,0.28) !important;
}
[data-testid="collapsedControl"] button:hover {
    border-color:rgba(255,140,50,0.6) !important;
}
.block-container { padding:3.75rem 3rem 4rem; max-width:1200px; }

/* Sidebar */
[data-testid="stSidebar"] { background:#0d0d14 !important; border-right:1px solid rgba(255,140,50,0.1) !important; }
[data-testid="stSidebar"] * { color:#e8e4dc; }
.sidebar-logo { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:800; color:#f0ebe0; letter-spacing:-0.02em; padding:0.5rem 0 1.2rem; }
.sidebar-logo span { color:#ff8c32; }
.sidebar-section-label { font-family:'DM Mono',monospace; font-size:0.62rem; letter-spacing:0.2em; text-transform:uppercase; color:#404040; margin:1rem 0 0.6rem; padding-bottom:0.4rem; border-bottom:1px solid rgba(255,255,255,0.05); }
.history-item { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.06); border-radius:10px; padding:0.7rem 0.9rem; margin-bottom:0.4rem; }
.history-item.active { border-color:rgba(255,140,50,0.5); background:rgba(255,140,50,0.07); }
.history-item-title { font-family:'DM Sans',sans-serif; font-size:0.82rem; font-weight:500; color:#e8e4dc; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.history-item-meta  { font-family:'DM Mono',monospace; font-size:0.6rem; color:#505050; margin-top:0.2rem; }
.history-item-badge { font-family:'DM Mono',monospace; font-size:0.55rem; color:#ff8c32; float:right; }

/* Hero */
.hero { text-align:center; padding:3.5rem 0 2.5rem; }
.hero-eyebrow { font-family:'DM Mono',monospace; font-size:0.7rem; font-weight:500; letter-spacing:0.25em; text-transform:uppercase; color:#ff8c32; margin-bottom:1rem; opacity:0.9; }
.hero h1 { font-family:'Syne',sans-serif; font-size:clamp(2.8rem,6vw,5rem); font-weight:800; line-height:1.0; letter-spacing:-0.03em; color:#f0ebe0; margin:0 0 1rem; }
.hero h1 span { color:#ff8c32; }
.hero-sub { font-size:1.05rem; font-weight:300; color:#a09890; max-width:520px; margin:0 auto; line-height:1.65; }
.divider { height:1px; background:linear-gradient(90deg,transparent,rgba(255,140,50,0.3),transparent); margin:2rem 0; }

/* Inputs */
.stTextInput > div > div > input { background:#1c1c24 !important; border:1px solid rgba(255,140,50,0.35) !important; border-radius:10px !important; color:#f0ebe0 !important; font-family:'DM Sans',sans-serif !important; font-size:1rem !important; padding:0.75rem 1rem !important; transition:border-color 0.2s,box-shadow 0.2s !important; }
.stTextInput > div > div > input:focus { border-color:#ff8c32 !important; box-shadow:0 0 0 3px rgba(255,140,50,0.12) !important; }
.stTextInput > label,.stTextArea > label { font-family:'DM Mono',monospace !important; font-size:0.72rem !important; letter-spacing:0.15em !important; text-transform:uppercase !important; color:#ff8c32 !important; font-weight:500 !important; }
.stTextArea textarea { background:#1c1c24 !important; border:1px solid rgba(255,140,50,0.25) !important; border-radius:10px !important; color:#f0ebe0 !important; font-family:'DM Sans',sans-serif !important; font-size:0.95rem !important; }

/* Buttons */
.stButton > button { background:linear-gradient(135deg,#ff8c32 0%,#ff5a1a 100%) !important; color:#0a0a0f !important; font-family:'Syne',sans-serif !important; font-weight:700 !important; font-size:0.95rem !important; letter-spacing:0.04em !important; border:none !important; border-radius:10px !important; padding:0.7rem 2.2rem !important; cursor:pointer !important; transition:transform 0.15s,box-shadow 0.15s,opacity 0.15s !important; box-shadow:0 4px 20px rgba(255,140,50,0.3) !important; width:100%; }
.stButton > button:hover  { transform:translateY(-2px) !important; box-shadow:0 8px 28px rgba(255,140,50,0.4) !important; opacity:0.95 !important; }
.stButton > button:active { transform:translateY(0) !important; }

/* Pipeline */
.step-card { background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:1.5rem 1.8rem; margin-bottom:1.2rem; position:relative; overflow:hidden; transition:border-color 0.3s; }
.step-card.active { border-color:rgba(255,140,50,0.4); background:rgba(255,140,50,0.04); }
.step-card.done   { border-color:rgba(80,200,120,0.3);  background:rgba(80,200,120,0.03); }
.step-card::before { content:''; position:absolute; left:0; top:0; bottom:0; width:3px; border-radius:14px 0 0 14px; background:rgba(255,255,255,0.05); transition:background 0.3s; }
.step-card.active::before { background:#ff8c32; }
.step-card.done::before   { background:#50c878; }
.step-header { display:flex; align-items:center; gap:0.8rem; margin-bottom:0.3rem; }
.step-num  { font-family:'DM Mono',monospace; font-size:0.68rem; font-weight:500; letter-spacing:0.15em; color:#ff8c32; opacity:0.7; }
.step-title { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; color:#f0ebe0; }
.step-status { margin-left:auto; font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.1em; }
.status-waiting { color:#555; } .status-running { color:#ff8c32; } .status-done { color:#50c878; }

/* Result panels */
.result-panel { background:rgba(255,255,255,0.025); border:1px solid rgba(255,255,255,0.07); border-radius:14px; padding:1.8rem 2rem; margin-top:1rem; margin-bottom:1.5rem; }
.result-panel-title { font-family:'DM Mono',monospace; font-size:0.7rem; font-weight:500; letter-spacing:0.2em; text-transform:uppercase; color:#ff8c32; margin-bottom:1rem; padding-bottom:0.7rem; border-bottom:1px solid rgba(255,140,50,0.15); }
.result-content { font-size:0.92rem; line-height:1.8; color:#e8e4dc; white-space:pre-wrap; font-family:'DM Sans',sans-serif; }
.report-panel   { background:rgba(255,255,255,0.025); border:1px solid rgba(255,140,50,0.2); border-radius:16px; padding:2rem 2.5rem; margin-top:1rem; }
.feedback-panel { background:rgba(255,255,255,0.025); border:1px solid rgba(80,200,120,0.2); border-radius:16px; padding:2rem 2.5rem; margin-top:1rem; }
.panel-label { font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; margin-bottom:1.2rem; padding-bottom:0.7rem; }
.panel-label.orange { color:#ff8c32; border-bottom:1px solid rgba(255,140,50,0.15); }
.panel-label.green  { color:#50c878; border-bottom:1px solid rgba(80,200,120,0.15); }

/* Action bar */
.action-bar       { background:rgba(255,255,255,0.02); border:1px solid rgba(255,140,50,0.12); border-radius:14px; padding:1.5rem 2rem; margin-top:1.5rem; }
.action-bar-title { font-family:'DM Mono',monospace; font-size:0.7rem; letter-spacing:0.2em; text-transform:uppercase; color:#ff8c32; margin-bottom:1rem; }

/* Chat */
.chat-container { background:rgba(255,255,255,0.02); border:1px solid rgba(255,140,50,0.2); border-radius:16px; overflow:hidden; margin-top:1rem; }
.chat-header { background:rgba(255,140,50,0.07); border-bottom:1px solid rgba(255,140,50,0.15); padding:0.9rem 1.5rem; display:flex; align-items:center; gap:0.6rem; }
.chat-header-title { font-family:'Syne',sans-serif; font-weight:700; font-size:0.95rem; color:#f0ebe0; }
.chat-header-sub   { font-family:'DM Mono',monospace; font-size:0.62rem; color:#ff8c32; letter-spacing:0.1em; margin-left:auto; }
.chat-messages { padding:1.2rem 1.5rem; }
.chat-bubble { margin-bottom:1rem; display:flex; flex-direction:column; }
.chat-bubble.user      { align-items:flex-end; }
.chat-bubble.assistant { align-items:flex-start; }
.chat-bubble-inner { max-width:80%; padding:0.7rem 1rem; border-radius:12px; font-size:0.88rem; line-height:1.65; font-family:'DM Sans',sans-serif; }
.chat-bubble.user .chat-bubble-inner      { background:rgba(255,140,50,0.15); border:1px solid rgba(255,140,50,0.25); color:#f0ebe0; border-bottom-right-radius:3px; }
.chat-bubble.assistant .chat-bubble-inner { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); color:#e8e4dc; border-bottom-left-radius:3px; }
.chat-role-label { font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.12em; text-transform:uppercase; margin-bottom:0.3rem; }
.chat-bubble.user .chat-role-label      { color:#ff8c32; }
.chat-bubble.assistant .chat-role-label { color:#50c878; }
.chat-chip { background:rgba(255,255,255,0.04); border:1px solid rgba(255,255,255,0.08); border-radius:20px; padding:0.2rem 0.7rem; font-size:0.72rem; color:#a09890; font-family:'DM Sans',sans-serif; }

/* Misc */
.stSpinner > div { color:#ff8c32 !important; }
.report-panel p,.report-panel li,.report-panel h1,.report-panel h2,.report-panel h3,
.feedback-panel p,.feedback-panel li,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li { color:#e8e4dc !important; }
details summary { font-family:'DM Mono',monospace !important; font-size:0.75rem !important; color:#a09890 !important; letter-spacing:0.1em !important; cursor:pointer; }
.section-heading { font-family:'Syne',sans-serif; font-size:1.3rem; font-weight:700; color:#f0ebe0; margin:2rem 0 1rem; }
.notice { font-family:'DM Mono',monospace; font-size:0.72rem; color:#605850; text-align:center; margin-top:3rem; letter-spacing:0.08em; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def step_card(num, title, state, desc=""):
    sm = {"waiting":("WAITING","status-waiting"),"running":("● RUNNING","status-running"),"done":("✓ DONE","status-done")}
    label, cls = sm.get(state,("",""))
    cc = {"running":"active","done":"done"}.get(state,"")
    st.markdown(f"""<div class="step-card {cc}"><div class="step-header">
        <span class="step-num">{num}</span><span class="step-title">{title}</span>
        <span class="step-status {cls}">{label}</span></div>
        {"<div style='font-size:0.82rem;color:#706860;margin-top:0.3rem;'>"+desc+"</div>" if desc else ""}
    </div>""", unsafe_allow_html=True)

def run_action(system_prompt, content):
    p = ChatPromptTemplate.from_messages([("system", system_prompt),("human","{content}")])
    return (p | llm | StrOutputParser()).invoke({"content": content})


# ── Session state init ────────────────────────────────────────────────────────
for key, default in [
    ("results",           {}),
    ("running",           False),
    ("done",              False),
    ("action_result",     None),
    ("action_label",      ""),
    ("action_running",    False),
    ("chat_open",         False),
    ("chat_history",      []),
    ("active_session_id", None),
    ("_current_topic",    ""),
    ("_pending_action",   ""),
    ("_pending_chat_q",   ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Research<span>Mind</span></div>', unsafe_allow_html=True)

    if st.button("＋  New Research", use_container_width=True, key="new_btn"):
        st.session_state.update({
            "results": {}, "running": False, "done": False,
            "action_result": None, "action_label": "", "action_running": False,
            "chat_open": False, "chat_history": [], "active_session_id": None,
            "topic_input": "", "_current_topic": "", "_pending_action": "", "_pending_chat_q": "",
        })
        st.rerun()

    all_sessions = get_all_sessions()

    if all_sessions:
        st.markdown('<div class="sidebar-section-label">Research History</div>', unsafe_allow_html=True)
        for sess in all_sessions:
            sid       = sess.id
            title     = (sess.topic or "Untitled")[:36]
            date_str  = fmt_date(sess.ts)
            n_chat    = len(sess.messages)
            is_active = sid == st.session_state.active_session_id
            meta = f"{date_str} · {n_chat} msg" if n_chat else date_str
            button_label = f"● {title}" if is_active else f"🔬 {title}"

            col_load, col_del = st.columns([5, 1])
            with col_load:
                if st.button(button_label, key=f"load_{sid}", use_container_width=True, help=meta):
                    loaded = get_session(sid)
                    if loaded:
                        restore_session(loaded)
                        st.rerun()
                st.markdown(f'<div class="history-item-meta">{meta}</div>', unsafe_allow_html=True)
            with col_del:
                if st.button("🗑", key=f"del_{sid}", help="Delete"):
                    delete_session(sid)
                    if st.session_state.active_session_id == sid:
                        st.session_state.update({
                            "results": {}, "done": False,
                            "chat_history": [], "active_session_id": None,
                            "topic_input": "", "_current_topic": "",
                        })
                    st.rerun()
    else:
        st.markdown("""<div style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#333;
            text-align:center;padding:2rem 0;letter-spacing:0.1em;">NO HISTORY YET</div>""",
            unsafe_allow_html=True)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">Four specialized AI agents collaborate — searching, scraping, writing,
    and critiquing — to deliver a polished research report on any topic.</p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)


# ── Layout ────────────────────────────────────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    topic = st.text_input("Research Topic",
        placeholder="e.g. Quantum computing breakthroughs in 2025",
        key="topic_input", label_visibility="visible")
    run_btn = st.button("⚡  Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem;">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#605850;letter-spacing:0.1em;">TRY →</span>""",
        unsafe_allow_html=True)
    for ex in ["LLM agents 2025", "CRISPR gene editing", "Fusion energy progress"]:
        st.markdown(f"""<span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
            border-radius:6px;padding:0.25rem 0.7rem;font-size:0.75rem;color:#a09890;
            font-family:'DM Sans',sans-serif;cursor:default;">{ex}</span>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Pipeline</div>', unsafe_allow_html=True)
    r = st.session_state.results
    def s(step):
        if not r: return "waiting"
        if step in r: return "done"
        if st.session_state.running:
            for k in ["search","reader","writer","critic"]:
                if k not in r: return "running" if k == step else "waiting"
        return "waiting"
    step_card("01","Search Agent", s("search"),"Gathers recent web information")
    step_card("02","Reader Agent", s("reader"),"Scrapes & extracts deep content")
    step_card("03","Writer Chain", s("writer"),"Drafts the full research report")
    step_card("04","Critic Chain", s("critic"),"Reviews & scores the report")


# ── Trigger ───────────────────────────────────────────────────────────────────
if run_btn:
    if not topic.strip():
        st.warning("Please enter a research topic first.")
    else:
        st.session_state.update({
            "results": {}, "running": True, "done": False,
            "action_result": None, "action_label": "", "action_running": False,
            "chat_open": False, "chat_history": [],
            "active_session_id": None, "_current_topic": topic.strip(),
            "_pending_action": "", "_pending_chat_q": "",
        })
        st.rerun()


# ── Execute pipeline ──────────────────────────────────────────────────────────
if st.session_state.running and not st.session_state.done:
    topic_val = st.session_state.topic_input

    with st.spinner("Running research pipeline..."):
        result = run_research(topic_val)
    st.session_state.results = dict(result["state"])
    st.session_state.active_session_id = result["session_id"]
    st.session_state.running = False
    st.session_state.done    = True
    st.rerun()


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    if r.get("query_type") == "SIMPLE":
        st.markdown('<div class="section-heading">💬 Answer</div>', unsafe_allow_html=True)
        st.markdown(r.get("direct_answer",""))
    else:
        st.markdown('<div class="section-heading">Results</div>', unsafe_allow_html=True)
        if "search" in r:
            with st.expander("🔍 Search Results (raw)", expanded=False):
                st.markdown(f'<div class="result-panel"><div class="result-panel-title">Search Agent Output</div><div class="result-content">{r["search"]}</div></div>', unsafe_allow_html=True)
        if "reader" in r:
            with st.expander("📄 Scraped Content (raw)", expanded=False):
                st.markdown(f'<div class="result-panel"><div class="result-panel-title">Reader Agent Output</div><div class="result-content">{r["reader"]}</div></div>', unsafe_allow_html=True)
        if "writer" in r:
            st.markdown('<div class="report-panel"><div class="panel-label orange">📝 Final Research Report</div>', unsafe_allow_html=True)
            st.markdown(r["writer"])
            st.markdown('</div>', unsafe_allow_html=True)
            st.download_button("⬇  Download Report (.md)", data=r["writer"],
                file_name=f"research_report_{int(time.time())}.md", mime="text/markdown")
        if "critic" in r:
            st.markdown('<div class="feedback-panel"><div class="panel-label green">🧐 Critic Feedback</div>', unsafe_allow_html=True)
            st.markdown(r["critic"])
            st.markdown('</div>', unsafe_allow_html=True)

    # ── Action bar ────────────────────────────────────────────────────────────
    base_content = r.get("writer") or r.get("direct_answer","")

    if base_content:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown('<div class="action-bar"><div class="action-bar-title">⚡ What do you want to do next?</div>', unsafe_allow_html=True)

        locked = st.session_state.action_running

        if locked:
            st.markdown("""<div style="background:rgba(255,80,30,0.08);border:1px solid rgba(255,80,30,0.3);
                border-radius:10px;padding:0.7rem 1.2rem;margin-bottom:1rem;
                font-family:'DM Mono',monospace;font-size:0.72rem;color:#ff6040;letter-spacing:0.08em;">
                ⏳ ACTION RUNNING — stop it before starting a new one</div>""", unsafe_allow_html=True)
            if st.button("⛔  Stop current action", key="stop_action"):
                st.session_state.action_running  = False
                st.session_state._pending_action = ""
                st.session_state._pending_chat_q = ""
                st.rerun()

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🧒 Simplify", use_container_width=True, key="btn_simplify", disabled=locked):
                st.session_state.action_running  = True
                st.session_state.action_result   = None
                st.session_state._pending_action = "simplify"
                st.rerun()
        with col2:
            if st.button("🐦 Twitter/X thread", use_container_width=True, key="btn_thread", disabled=locked):
                st.session_state.action_running  = True
                st.session_state.action_result   = None
                st.session_state._pending_action = "thread"
                st.rerun()
        with col3:
            if st.button("📊 Key bullet points", use_container_width=True, key="btn_bullets", disabled=locked):
                st.session_state.action_running  = True
                st.session_state.action_result   = None
                st.session_state._pending_action = "bullets"
                st.rerun()

        chat_lbl = "💬  Close Chat" if st.session_state.chat_open else "💬  Ask / Chat with Report"
        if st.button(chat_lbl, use_container_width=True, key="btn_chat", disabled=locked):
            st.session_state.chat_open = not st.session_state.chat_open
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        # Execute pending quick action
        pending = st.session_state._pending_action
        if st.session_state.action_running and not st.session_state.chat_open and pending:
            action_map = {
                "simplify": ("Rewrite this content so a 10-year-old can understand it. Use simple words and fun analogies.", "🧒 Simplified Version"),
                "thread":   ("Convert this into a compelling Twitter/X thread. Number each tweet. Max 280 chars each. Start with a strong hook.", "🐦 Twitter/X Thread"),
                "bullets":  ("Extract the 5–7 most important insights as concise, numbered bullet points.", "📊 Key Bullet Points"),
            }
            sys_p, lbl = action_map[pending]
            with st.spinner(f"Running: {lbl}…"):
                st.session_state.action_result  = run_action(sys_p, base_content)
                st.session_state.action_label   = lbl
            st.session_state.action_running  = False
            st.session_state._pending_action = ""
            st.rerun()

        if st.session_state.get("action_result"):
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-panel"><div class="result-panel-title">{st.session_state.action_label}</div></div>', unsafe_allow_html=True)
            st.markdown(st.session_state.action_result)
            if st.button("🗑 Clear result", key="clear_action"):
                st.session_state.action_result = None
                st.session_state.action_label  = ""
                st.rerun()

        # ── Chat panel ────────────────────────────────────────────────────────
        if st.session_state.chat_open:
            st.markdown("""<div class="chat-container"><div class="chat-header">
                <span style="font-size:1.1rem;">🤖</span>
                <span class="chat-header-title">Research Assistant</span>
                <span class="chat-header-sub">MULTI-TURN · KNOWS YOUR REPORT</span>
            </div></div>""", unsafe_allow_html=True)

            if not st.session_state.chat_history:
                st.markdown("""<div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin:0.8rem 0;">
                    <span style="font-family:'DM Mono',monospace;font-size:0.62rem;color:#505050;letter-spacing:0.1em;align-self:center;">TRY →</span>
                    <span class="chat-chip">Translate to Hindi</span>
                    <span class="chat-chip">Write an email pitch</span>
                    <span class="chat-chip">Find weaknesses</span>
                    <span class="chat-chip">Explain point 2 in detail</span>
                </div>""", unsafe_allow_html=True)

            if st.session_state.chat_history:
                bubbles = '<div class="chat-messages">'
                for msg in st.session_state.chat_history:
                    role    = msg["role"]
                    content = msg["content"].replace("<","&lt;").replace(">","&gt;")
                    lbl_    = "You" if role == "user" else "ResearchMind"
                    bubbles += f'<div class="chat-bubble {role}"><div class="chat-role-label">{lbl_}</div><div class="chat-bubble-inner">{content}</div></div>'
                bubbles += "</div>"
                st.markdown(bubbles, unsafe_allow_html=True)
            else:
                st.markdown("""<div style="text-align:center;padding:1.5rem;color:#404040;
                    font-family:'DM Mono',monospace;font-size:0.72rem;letter-spacing:0.1em;">
                    ASK ANYTHING · ONE-OFF OR MULTI-TURN</div>""", unsafe_allow_html=True)

            col_q, col_send = st.columns([6,1])
            with col_q:
                user_q = st.text_input("Message", key="chat_input",
                    placeholder='"Translate to Hindi" · "Find weaknesses" · "Tell me more about point 3"',
                    label_visibility="collapsed", disabled=locked)
            with col_send:
                send = st.button("Send ➤", key="chat_send", use_container_width=True, disabled=locked)

            if send and user_q.strip():
                st.session_state.action_running  = True
                st.session_state._pending_chat_q = user_q.strip()
                st.rerun()

            # Execute chat turn
            if st.session_state.action_running and st.session_state.chat_open and st.session_state._pending_chat_q:
                pq = st.session_state._pending_chat_q
                with st.spinner("Thinking…"):
                    response = send_chat_message(
                        session_id=st.session_state.active_session_id,
                        question=pq,
                        report=base_content,
                    )
                st.session_state.chat_history.append({"role":"user",      "content": pq})
                st.session_state.chat_history.append({"role":"assistant",  "content": response})
                st.session_state.action_running   = False
                st.session_state._pending_chat_q  = ""
                st.rerun()

            if st.session_state.chat_history:
                if st.button("🗑 Clear chat", key="clear_chat"):
                    st.session_state.chat_history = []
                    if st.session_state.active_session_id:
                        clear_chat_history(st.session_state.active_session_id)
                    st.rerun()


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">ResearchMind · Powered by LangChain multi-agent pipeline · Built with Streamlit</div>
""", unsafe_allow_html=True)
