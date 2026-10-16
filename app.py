"""AI Video Assistant — Streamlit UI (professional edition).

Run with:  streamlit run app.py
Put this file in the project root (next to `core/` and `utils/`) and copy the
`.streamlit/config.toml` folder alongside it for the theme.
"""
from pathlib import Path
import tempfile

import streamlit as st
from dotenv import load_dotenv

# Load credentials before importing modules that read them at import time.
load_dotenv(Path(__file__).with_name(".env"))

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.summarizer import summarize, generate_title
from core.extractor import (
    extract_action_items,
    extract_key_decisions,
    extract_questions,
)
from core.rag_engine import build_rag_chain, ask_question

# --------------------------------------------------------------------------- #
# Page config + styling
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="AI Video Assistant",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, button, input, textarea {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
}
/* Hide default chrome */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {visibility: hidden; height: 0;}
header[data-testid="stHeader"] {background: transparent;}
.block-container {padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1120px;}

/* Hero */
.hero {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 55%, #db2777 120%);
    border-radius: 20px; padding: 2.2rem 2.4rem; color: #fff; margin-bottom: 1.6rem;
    box-shadow: 0 10px 30px rgba(99,102,241,.25);
}
.hero .badge {
    display: inline-block; font-size: .7rem; font-weight: 700; letter-spacing: .12em;
    background: rgba(255,255,255,.18); padding: .3rem .7rem; border-radius: 999px; margin-bottom: .8rem;
}
.hero h1 {font-size: 2.3rem; font-weight: 800; margin: 0 0 .4rem 0; color: #fff; padding: 0;}
.hero p {margin: 0; font-size: 1.02rem; opacity: .92; max-width: 640px; line-height: 1.55;}

/* Feature cards (empty state) */
.feature {
    border: 1px solid rgba(128,128,128,.22); border-radius: 16px; padding: 1.3rem 1.3rem 1.1rem;
    height: 100%; background: rgba(128,128,128,.05); transition: all .2s ease;
}
.feature:hover {transform: translateY(-3px); border-color: #6366f1; box-shadow: 0 8px 22px rgba(99,102,241,.15);}
.feature .ico {font-size: 1.7rem; margin-bottom: .5rem;}
.feature h4 {margin: 0 0 .35rem 0; font-size: 1.02rem; font-weight: 700;}
.feature p {margin: 0; font-size: .9rem; opacity: .75; line-height: 1.5;}

/* Result header */
.result-head {
    border-left: 5px solid #6366f1; padding: .4rem 0 .4rem 1rem; margin: .2rem 0 1.2rem 0;
}
.result-head .label {font-size: .72rem; font-weight: 700; letter-spacing: .1em; color: #6366f1;}
.result-head h2 {margin: .15rem 0 0 0; font-size: 1.7rem; font-weight: 800; padding: 0;}

/* Metrics */
div[data-testid="stMetric"] {
    border: 1px solid rgba(128,128,128,.22); background: rgba(128,128,128,.05);
    padding: .9rem 1.1rem; border-radius: 14px;
}
div[data-testid="stMetricLabel"] p {font-size: .78rem; font-weight: 600; opacity: .7; text-transform: uppercase; letter-spacing: .05em;}
div[data-testid="stMetricValue"] {font-size: 1.5rem; font-weight: 700;}

/* Tabs */
button[data-baseweb="tab"] {font-weight: 600; padding: .6rem 1rem;}
div[data-baseweb="tab-highlight"] {background-color: #6366f1 !important; height: 3px; border-radius: 3px;}
div[data-baseweb="tab-list"] {gap: .25rem; border-bottom: 1px solid rgba(128,128,128,.2);}

/* Bordered containers act as cards */
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}

/* Buttons */
.stButton > button, .stDownloadButton > button {
    border-radius: 10px; font-weight: 600; transition: all .15s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed); border: none; color: #fff;
    box-shadow: 0 4px 14px rgba(99,102,241,.35);
}
.stButton > button[kind="primary"]:hover {transform: translateY(-1px); box-shadow: 0 6px 18px rgba(99,102,241,.45);}

/* Sidebar */
section[data-testid="stSidebar"] {border-right: 1px solid rgba(128,128,128,.18);}
.side-brand {font-size: 1.15rem; font-weight: 800; margin-bottom: .2rem;}
.side-sub {font-size: .8rem; opacity: .65; margin-bottom: 1rem;}

/* Pipeline stepper */
.steps {border: 1px solid rgba(128,128,128,.22); border-radius: 16px; padding: 1.2rem 1.4rem; background: rgba(128,128,128,.05);}
.step {display: flex; align-items: center; gap: .8rem; padding: .45rem 0; font-size: .95rem;}
.step .dot {
    width: 26px; height: 26px; border-radius: 50%; display: flex; align-items: center; justify-content: center;
    font-size: .75rem; font-weight: 700; border: 2px solid rgba(128,128,128,.35); flex-shrink: 0;
}
.step.done {opacity: .85;}
.step.done .dot {background: #10b981; border-color: #10b981; color: #fff;}
.step.active {font-weight: 700;}
.step.active .dot {border-color: #6366f1; color: #6366f1; animation: pulse 1.2s infinite;}
.step.todo {opacity: .45;}
@keyframes pulse {0% {box-shadow: 0 0 0 0 rgba(99,102,241,.5);} 70% {box-shadow: 0 0 0 9px rgba(99,102,241,0);} 100% {box-shadow: 0 0 0 0 rgba(99,102,241,0);}}

.footer {text-align: center; font-size: .78rem; opacity: .5; margin-top: 3rem;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def html(markup: str) -> None:
    """Render raw HTML safely (strips indentation so markdown doesn't treat it as code)."""
    flat = "".join(line.strip() for line in markup.strip().splitlines())
    st.markdown(flat, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
for key, value in {
    "result": None,
    "messages": [],
    "language": "english",
    "source_label": "",
    "pending_question": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

STEPS = [
    "Preparing audio",
    "Transcribing speech",
    "Writing title & summary",
    "Extracting insights",
    "Building chat index",
]

SUGGESTED = [
    "Give me a 3-line recap",
    "What were the main decisions?",
    "Who owns which task?",
    "What is still unresolved?",
]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def as_markdown(value) -> str:
    """Render str / list / dict outputs from the extractors as markdown."""
    if not value:
        return "_Nothing found._"
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(f"- **{k}**: {v}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "\n".join(f"- {item}" for item in value)
    return str(value)


def build_report(result: dict) -> str:
    return (
        f"# {result['title']}\n\n"
        f"## Summary\n{as_markdown(result['summary'])}\n\n"
        f"## Action Items\n{as_markdown(result['action_items'])}\n\n"
        f"## Key Decisions\n{as_markdown(result['key_decisions'])}\n\n"
        f"## Open Questions\n{as_markdown(result['open_questions'])}\n\n"
        f"## Transcript\n{result['transcript']}\n"
    )


def render_steps(placeholder, current: int) -> None:
    rows = []
    for i, name in enumerate(STEPS):
        if i < current:
            state, mark = "done", "✓"
        elif i == current:
            state, mark = "active", str(i + 1)
        else:
            state, mark = "todo", str(i + 1)
        rows.append(f'<div class="step {state}"><div class="dot">{mark}</div><div>{name}</div></div>')
    flat = "".join(rows)
    placeholder.markdown(f'<div class="steps">{flat}</div>', unsafe_allow_html=True)


def run_pipeline_with_progress(source: str, language: str) -> dict:
    """Same steps as run_pipeline(), but with a live stepper + progress bar."""
    steps_box = st.empty()
    bar = st.progress(0)

    def advance(i: int) -> None:
        render_steps(steps_box, i)
        bar.progress(min(i / len(STEPS), 1.0))

    advance(0)
    chunks = process_input(source)

    advance(1)
    transcript = transcribe_all(chunks, language)
    if not transcript:
        steps_box.empty()
        bar.empty()
        raise RuntimeError("No speech was detected in the supplied audio.")

    advance(2)
    title = generate_title(transcript)
    summary = summarize(transcript)

    advance(3)
    action_items = extract_action_items(transcript)
    decisions = extract_key_decisions(transcript)
    questions = extract_questions(transcript)

    advance(4)
    rag_chain = build_rag_chain(transcript)

    render_steps(steps_box, len(STEPS))
    bar.progress(1.0)
    steps_box.empty()
    bar.empty()

    return {
        "title": title,
        "transcript": transcript,
        "summary": summary,
        "action_items": action_items,
        "key_decisions": decisions,
        "open_questions": questions,
        "rag_chain": rag_chain,
    }


def answer_question(question: str) -> None:
    st.session_state.messages.append({"role": "user", "content": question})
    try:
        answer = ask_question(st.session_state.result["rag_chain"], question)
    except Exception as exc:
        answer = f"⚠️ Sorry, something went wrong: {exc}"
    st.session_state.messages.append({"role": "assistant", "content": answer})


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    html('<div class="side-brand">🎬 AI Video Assistant</div>'
         '<div class="side-sub">Recordings → insights → answers</div>')

    input_type = st.radio("Source", ["YouTube URL", "Upload file"], horizontal=True)

    youtube_url, uploaded = "", None
    if input_type == "YouTube URL":
        youtube_url = st.text_input("Video link", placeholder="https://youtu.be/…")
    else:
        uploaded = st.file_uploader(
            "Audio / video file",
            type=["mp3", "wav", "m4a", "mp4", "mkv", "webm", "ogg", "flac"],
        )

    language = st.selectbox("Spoken language", ["english", "hinglish"])

    can_run = bool(youtube_url.strip()) if input_type == "YouTube URL" else uploaded is not None
    analyze = st.button("🚀  Analyze recording", type="primary", use_container_width=True, disabled=not can_run)

    if st.session_state.result:
        st.divider()
        if st.button("↺  Start over", use_container_width=True):
            st.session_state.result = None
            st.session_state.messages = []
            st.rerun()

    st.divider()
    st.caption("Tip: shorter, clearer audio gives better transcripts and more accurate answers.")

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
html(
    """
    <div class="hero">
        <div class="badge">AI POWERED</div>
        <h1>Turn any video into clear, actionable insights</h1>
        <p>Transcribe meetings, lectures and YouTube videos — then get summaries, action items,
        decisions, and a chat assistant that answers from the transcript.</p>
    </div>
    """
)

# --------------------------------------------------------------------------- #
# Run pipeline
# --------------------------------------------------------------------------- #
if analyze:
    st.session_state.messages = []
    st.session_state.result = None
    tmp_path = None
    try:
        if input_type == "Upload file":
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            source, label = tmp_path, uploaded.name
        else:
            source, label = youtube_url.strip(), "YouTube"

        st.session_state.result = run_pipeline_with_progress(source, language)
        st.session_state.language = language
        st.session_state.source_label = label
    except Exception as exc:
        st.error(f"Could not process this input: {exc}")
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

result = st.session_state.result

# --------------------------------------------------------------------------- #
# Empty state
# --------------------------------------------------------------------------- #
if not result:
    cols = st.columns(3, gap="medium")
    features = [
        ("🎧", "Transcribe", "Accurate speech-to-text for English and Hinglish, from YouTube links or your own files."),
        ("🧠", "Understand", "Get a crisp summary plus action items, key decisions and open questions in seconds."),
        ("💬", "Ask anything", "Chat with the recording. Every answer is grounded in what was actually said."),
    ]
    for col, (ico, title, desc) in zip(cols, features):
        with col:
            html(f'<div class="feature"><div class="ico">{ico}</div><h4>{title}</h4><p>{desc}</p></div>')
    st.write("")
    st.info("👈 Add a YouTube link or upload a file in the sidebar, then click **Analyze recording**.")
    html('<div class="footer">Built with Streamlit · AI Video Assistant</div>')
    st.stop()

# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
words = len(result["transcript"].split())
read_min = max(1, round(words / 200))

html(f'<div class="result-head"><div class="label">ANALYSIS COMPLETE</div><h2>{result["title"]}</h2></div>')

m1, m2, m3, m4 = st.columns(4, gap="small")
m1.metric("Words", f"{words:,}")
m2.metric("Read time", f"{read_min} min")
m3.metric("Language", st.session_state.language.capitalize())
m4.metric("Source", (st.session_state.source_label[:14] + "…") if len(st.session_state.source_label) > 15 else st.session_state.source_label)

st.write("")
tabs = st.tabs(["📋 Summary", "✅ Action items", "🔑 Decisions", "❓ Open questions", "📝 Transcript", "💬 Chat"])

with tabs[0]:
    with st.container(border=True):
        st.markdown(as_markdown(result["summary"]))

with tabs[1]:
    with st.container(border=True):
        st.markdown(as_markdown(result["action_items"]))

with tabs[2]:
    with st.container(border=True):
        st.markdown(as_markdown(result["key_decisions"]))

with tabs[3]:
    with st.container(border=True):
        st.markdown(as_markdown(result["open_questions"]))

with tabs[4]:
    st.text_area("Transcript", result["transcript"], height=420, label_visibility="collapsed")
    st.download_button("⬇️  Download transcript (.txt)", result["transcript"], "transcript.txt", "text/plain")

with tabs[5]:
    if not st.session_state.messages:
        st.caption("Try one of these, or type your own question below:")
        cols = st.columns(2)
        for i, q in enumerate(SUGGESTED):
            if cols[i % 2].button(q, key=f"sugg_{i}", use_container_width=True):
                st.session_state.pending_question = q
                st.rerun()

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
            st.markdown(msg["content"])

    typed = st.chat_input("Ask a question about this recording…")
    question = typed or st.session_state.pop("pending_question", None)
    if question:
        with st.chat_message("user", avatar="🧑‍💻"):
            st.markdown(question)
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking…"):
                answer_question(question)
            st.markdown(st.session_state.messages[-1]["content"])

st.write("")
st.download_button(
    "⬇️  Download full report (.md)",
    build_report(result),
    "meeting_report.md",
    "text/markdown",
    type="primary",
)
html('<div class="footer">Built with Streamlit · AI Video Assistant</div>')