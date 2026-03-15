import os
import streamlit as st
import config
from rag import initialize_rag, get_retriever
from tools import get_all_tools
from agent import build_agent, chat

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{config.YOUR_NAME} – Digital Twin",
    page_icon="🤖",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🤖 {config.YOUR_NAME}'s Digital Twin</h1>
    <p>{config.YOUR_TAGLINE}</p>
    <small>Powered by Groq · LangChain · RAG</small>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Info")
    st.markdown(f"""
    👤 **{config.YOUR_NAME}**  
    {config.YOUR_TAGLINE}
    """)

    st.divider()
    st.subheader("🔧 Tools Available")
    st.markdown("""
    - 📋 **Resume Search** (RAG)
    - 🧮 **Calculator**
    - 🌐 **Web Search**
    - 📖 **Wikipedia**
    - 🌤️ **Weather**
    """)

    st.divider()
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Model: `{config.GROQ_MODEL}`")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

if "agent" not in st.session_state:
    st.session_state.agent = None

# ── Auto-initialize agent on startup ─────────────────────────────────────────
# This runs ONCE when the app first loads.
# It either loads the saved vector store OR builds from my_resume.pdf
# Either way — no visitor needs to upload anything!

if st.session_state.agent is None:
    with st.spinner("🤖 Waking up your Digital Twin..."):
        try:
            # Path to your resume PDF sitting in the project folder
            RESUME_PATH = "my_resume.pdf"

            # initialize_rag() will:
            #   - Load existing vector_store/ if it exists
            #   - OR build from RESUME_PATH if not
            vector_store = initialize_rag(pdf_path=RESUME_PATH)
            retriever    = get_retriever(vector_store)
            tools        = get_all_tools(retriever)
            st.session_state.agent = build_agent(tools)

            # Add greeting message on very first load
            if len(st.session_state.messages) == 0:
                greeting = (
                    f"Hi there! 👋 I'm {config.YOUR_NAME}'s Digital Twin.\n\n"
                    f"I know everything about {config.YOUR_NAME}'s background, "
                    f"skills, and experience. I can also help with calculations, "
                    f"web searches, weather, and more!\n\n"
                    f"What would you like to know?"
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": greeting
                })

        except FileNotFoundError:
            st.error(
                "❌ Resume PDF not found!\n\n"
                "Make sure `my_resume.pdf` is in the project folder."
            )
            st.stop()
        except Exception as e:
            st.error(f"❌ Failed to start: {e}")
            st.stop()

# ── Display chat history ──────────────────────────────────────────────────────
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input(
    "Ask me anything...",
    disabled=(st.session_state.agent is None),
):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chat(st.session_state.agent, prompt)
        st.markdown(response)

    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })
st.divider()
st.caption(
    "Built with ❤️ using LangChain · Groq · FAISS · Streamlit"
)
