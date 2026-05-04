import streamlit as st
import re
import os

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

# OPTIONAL (for local .env support)
from dotenv import load_dotenv
load_dotenv()

# -----------------------------
# 🔐 API KEYS
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY not found.")
    st.stop()

# ✅ Set HuggingFace token globally
if HF_API_KEY:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = HF_API_KEY
else:
    st.warning("⚠️ HuggingFace token not found (optional).")

# -----------------------------
# 🤖 LLM
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=GOOGLE_API_KEY
)

# -----------------------------
# 🎯 Extract Video ID
# -----------------------------
def extract_video_id(url):
    match = re.search(r"(?:v=|youtu\.be/)([\w-]+)", url)
    return match.group(1) if match else url

# -----------------------------
# 📥 Get Transcript
# -----------------------------
def get_transcript(video_id, max_chars=50000):
    data = YouTubeTranscriptApi.get_transcript(video_id)
    text = " ".join([t["text"] for t in data])
    return text[:max_chars]

# -----------------------------
# 🧠 Vector Store (Cached)
# -----------------------------
@st.cache_resource
def create_vectorstore(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=20
    )
    docs = splitter.create_documents([text])

    #embedding = HuggingFaceEmbeddings(
       # model_name="sentence-transformers/all-MiniLM-L6-v2",
        #model_kwargs={"device": "cpu"},
        #encode_kwargs={"normalize_embeddings": True}
    #)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embedding = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GOOGLE_API_KEY
)

    return FAISS.from_documents(docs, embedding)

# -----------------------------
# 💬 RAG Answer
# -----------------------------
def get_answer(query, vectorstore):
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant.
        Answer ONLY from the provided transcript context.
        If the context is insufficient, say "I don't know".

        Context:
        {context}

        Question:
        {question}
        """,
        input_variables=["context", "question"]
    )

    final_prompt = prompt.invoke({
        "context": context,
        "question": query
    })

    response = llm.invoke(final_prompt)
    return response.content

# -----------------------------
# 🎨 UI
# -----------------------------
st.set_page_config(page_title="YouTube RAG Chat", layout="wide")
st.title("📺 YouTube Transcript Chatbot (RAG)")

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------
# 🔗 Input Section
# -----------------------------
video_url = st.text_input("Enter YouTube URL or Video ID")

if st.button("Load Video"):
    try:
        with st.spinner("Processing video..."):
            video_id = extract_video_id(video_url)
            transcript = get_transcript(video_id)

            st.session_state.vectorstore = create_vectorstore(transcript)
            st.session_state.messages = []

        st.success("✅ Video processed! You can now chat.")

    except TranscriptsDisabled:
        st.error("❌ No captions available for this video.")
    except Exception:
        st.error("⚠️ Failed to process video. Try another one.")

# -----------------------------
# 💬 Chat Section
# -----------------------------
if st.session_state.vectorstore:

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about the video...")

    if user_input:
        st.session_state.messages.append({
            "role": "user",
            "content": user_input
        })

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = get_answer(user_input, st.session_state.vectorstore)
                st.markdown(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })
