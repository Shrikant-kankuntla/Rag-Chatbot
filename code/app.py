import streamlit as st
import re
import time
from datetime import datetime

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# ---------------- CONFIG ----------------
CHROMA_PATH = "db"

PROMPT_TEMPLATE = """
You are a helpful assistant.

Rules:
- Answer ONLY using the provided context
- If the answer is not clearly present, respond exactly:
  "Answer not found in the document."
- Do NOT guess
- Do NOT use outside knowledge
- Keep answers clear and concise

Chat History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""

# ---------------- STYLE (PRO LOOK) ----------------
st.set_page_config(page_title="RAG Assistant", page_icon="💬", layout="wide")

st.markdown("""
<style>
/* App background */
.stApp { background: #0f172a; }

/* Header */
.header {
  padding: 16px 18px; border-radius: 14px;
  background: linear-gradient(135deg, #111827, #1f2937);
  color: #e5e7eb; margin-bottom: 10px;
  box-shadow: 0 10px 25px rgba(0,0,0,0.25);
}

/* Chat bubbles */
.user-bubble {
  background: #2563eb; color: white;
  padding: 12px 16px; border-radius: 14px;
  margin: 6px 0; width: fit-content; max-width: 80%;
  margin-left: auto;
}
.bot-bubble {
  background: #1f2937; color: #e5e7eb;
  padding: 12px 16px; border-radius: 14px;
  margin: 6px 0; width: fit-content; max-width: 80%;
}

/* Sidebar */
section[data-testid="stSidebar"] {
  background: #020617;
}

/* Buttons */
.stButton button {
  border-radius: 10px; padding: 8px 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="header">
  <h2> RAG Assistant</h2>
  <p>Grounded answers • No hallucinations • Local AI (Ollama)</p>
</div>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def is_realtime_query(query):
    keywords = ["time", "date", "today", "current", "now", "year", "month", "clock"]
    return any(k in query.lower() for k in keywords)

def safe_highlight(text, answer):
    try:
        for word in set(answer.split()):
            if len(word) > 5:
                safe = re.escape(word)
                text = re.sub(f"({safe})", r"<mark>\1</mark>", text, flags=re.IGNORECASE)
        return text
    except:
        return text

def safe_retrieve(db, query):
    try:
        results = db.similarity_search_with_score(query, k=2)
        docs = [doc for doc, score in results if score < 0.8] or [doc for doc, _ in results[:2]]
        return docs
    except Exception as e:
        st.error(f"Retrieval error: {e}")
        return []

def format_chat_download(chat_history):
    lines = []
    for role, msg in chat_history:
        who = "User" if role == "user" else "Assistant"
        lines.append(f"{who}: {msg}")
    return "\n\n".join(lines)

# ---------------- LOAD ----------------
@st.cache_resource
def load_db():
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return Chroma(persist_directory=CHROMA_PATH, embedding_function=emb)

@st.cache_resource
def load_model():
    return ChatOllama(model="mistral", temperature=0, num_predict=120)

db = load_db()
model = load_model()

# ---------------- STATE ----------------
if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("🧭 Control Panel")

st.sidebar.markdown("**Recent Messages**")
for role, msg in st.session_state.chat_history[-8:]:
    icon = "🧑" if role == "user" else "🤖"
    st.sidebar.write(f"{icon} {msg[:40]}...")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.history = []
    st.rerun()

# Download chat
chat_text = format_chat_download(st.session_state.chat_history)
st.sidebar.download_button(
    label="📥 Download Chat (.txt)",
    data=chat_text.encode("utf-8"),
    file_name=f"rag_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
    mime="text/plain"
)

st.sidebar.caption("Tip: Ask document-based questions for best results.")

# ---------------- CHAT INPUT ----------------
query = st.chat_input("Ask something from your document...")

# ---------------- RENDER EXISTING CHAT ----------------
for role, msg in st.session_state.chat_history:
    if role == "user":
        st.markdown(f'<div class="user-bubble">{msg}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-bubble">{msg}</div>', unsafe_allow_html=True)

# ---------------- MAIN ----------------
if query:
    start = time.time()

    # show user
    st.session_state.chat_history.append(("user", query))
    st.markdown(f'<div class="user-bubble">{query}</div>', unsafe_allow_html=True)

    # ⏰ real-time block
    if is_realtime_query(query):
        resp = "⏰ Real-time information is not available in this system."
        st.session_state.chat_history.append(("assistant", resp))
        st.markdown(f'<div class="bot-bubble">{resp}</div>', unsafe_allow_html=True)
        st.stop()

    # retrieve
    docs = safe_retrieve(db, query)
    if not docs:
        resp = "Answer not found in the document."
        st.session_state.chat_history.append(("assistant", resp))
        st.markdown(f'<div class="bot-bubble">{resp}</div>', unsafe_allow_html=True)
        st.stop()

    context = "\n\n".join([d.page_content for d in docs])[:600]
    history = "\n".join(st.session_state.history[-4:])

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    final_prompt = prompt.format(context=context, question=query, history=history)

    # streaming
    placeholder = st.empty()
    full = ""

    try:
        for chunk in model.stream(final_prompt):
            if chunk.content:
                full += chunk.content
                placeholder.markdown(f'<div class="bot-bubble">{full}</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Model error: {e}")
        st.stop()

    answer = full.strip()

    # save
    st.session_state.history += [f"User: {query}", f"Bot: {answer}"]
    st.session_state.chat_history.append(("assistant", answer))

    # sources + highlight
    with st.expander("📄 Source Chunks"):
        for i, d in enumerate(docs):
            st.code(d.page_content[:300])

    with st.expander("🔍 Highlighted Context"):
        st.markdown(safe_highlight(context, answer), unsafe_allow_html=True)

    st.caption(f"⚡ Response time: {round(time.time() - start, 2)} sec")