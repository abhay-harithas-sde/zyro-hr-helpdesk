"""
Zyro Dynamics HR Help Desk — Streamlit Chatbot
RAG pipeline: FAISS + HuggingFace embeddings + Groq LLM + LangChain
"""
import os
import streamlit as st
from pathlib import Path

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.main-header{background:linear-gradient(135deg,#1e3a5f,#2d6a9f);color:white;padding:1.5rem 2rem;border-radius:12px;margin-bottom:1.5rem;text-align:center}
.main-header h1{margin:0;font-size:1.8rem}
.main-header p{margin:.3rem 0 0;opacity:.85;font-size:.95rem}
.source-card{background:#f0f7ff;border-left:4px solid #2d6a9f;border-radius:6px;padding:.6rem 1rem;margin:.4rem 0;font-size:.82rem;color:#333}
.oos-card{background:#fff3cd;border-left:4px solid #ffc107;border-radius:6px;padding:.6rem 1rem;margin:.4rem 0}
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PDF_DIR          = Path(__file__).parent / "pdfs"
CHUNK_SIZE       = 800
CHUNK_OVERLAP    = 150
TOP_K            = 5
EMBEDDING_MODEL  = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL        = "llama-3.3-70b-versatile"

OOS_KEYWORDS = [
    "stock price","share price","revenue","profit","quarterly","market cap",
    "investment","trading","ipo","dividend","competitor","weather","cricket",
    "sports","movie","recipe","cook","election","politics","religion",
    "personal loan","credit card","net worth","apply for a job",
    "recruitment process","hiring process","product features","compare to salesforce",
    "acruxcrm features","leave policy is at zoho","leave policy is at freshworks",
    "performing financially","revenue last year",
]

REFUSAL = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. Your question appears to be "
    "outside the scope of our HR knowledge base."
)

# ── Build RAG pipeline (cached) ───────────────────────────────────────────────
@st.cache_resource(show_spinner="Building HR knowledge base… ⏳")
def build_pipeline():
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        st.error("⚠️ GROQ_API_KEY not set. Add it in Streamlit Secrets.")
        st.stop()

    # Load PDFs
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = Path(doc.metadata.get("source", "")).name

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # Embed + FAISS
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever   = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.6},
    )

    # LLM
    llm = ChatGroq(model=LLM_MODEL, api_key=groq_key, temperature=0.1, max_tokens=512)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer ONLY based on the retrieved HR policy context below.
- Be accurate, concise and complete.
- If the context lacks information, say so honestly. Do NOT fabricate.
- Format lists clearly when appropriate.

Context:
{context}"""),
        ("human", "{question}"),
    ])

    def fmt(docs):
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('source_file','?')} | p{d.metadata.get('page',0)+1}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        RunnablePassthrough.assign(context=lambda x: fmt(retriever.invoke(x["question"])))
        | prompt | llm | StrOutputParser()
    )

    return retriever, chain, len(documents), len(chunks)

# ── Helpers ───────────────────────────────────────────────────────────────────
def ask(question: str, retriever, chain) -> dict:
    q = question.lower()
    if any(kw in q for kw in OOS_KEYWORDS):
        return {"answer": REFUSAL, "sources": [], "oos": True}

    docs   = retriever.invoke(question)
    answer = chain.invoke({"question": question})

    oos_phrases = [
        "does not contain information","does not provide information",
        "context does not contain","cannot answer","no information available",
        "not available in the provided","outside the scope",
    ]
    if any(p in answer.lower() for p in oos_phrases):
        return {"answer": REFUSAL, "sources": [], "oos": True}

    seen, sources = set(), []
    for d in docs:
        k = (d.metadata.get("source_file",""), d.metadata.get("page",""))
        if k not in seen:
            seen.add(k)
            sources.append({
                "file":    d.metadata.get("source_file", "Unknown"),
                "page":    d.metadata.get("page", 0) + 1,
                "snippet": d.page_content[:200].replace("\n", " ") + "…",
            })

    return {"answer": answer, "sources": sources, "oos": False}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Zyro Dynamics")
    st.markdown("**HR Help Desk Assistant**")
    st.divider()

    retriever, chain, n_docs, n_chunks = build_pipeline()
    st.success("✅ Knowledge base ready")
    st.markdown(f"- 📄 **{n_docs}** pages loaded")
    st.markdown(f"- 🔢 **{n_chunks:,}** chunks indexed")
    st.divider()

    st.markdown("### 📚 Policy Documents")
    for doc in [
        "Company Profile","Employee Handbook","Leave Policy",
        "Work From Home Policy","Code of Conduct",
        "Performance Review Policy","Compensation & Benefits",
        "IT & Data Security","POSH Policy",
        "Onboarding & Separation","Travel & Expense",
    ]:
        st.markdown(f"• {doc}")

    st.divider()
    st.markdown("### 💡 Try These")
    samples = [
        "How many earned leaves per year?",
        "What is the WFH policy?",
        "How does the PIP process work?",
        "What is maternity leave entitlement?",
    ]
    for s in samples:
        if st.button(s, key=s, use_container_width=True):
            st.session_state["prefill"] = s

    st.divider()
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()

# ── Main ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏢 Zyro Dynamics HR Help Desk</h1>
  <p>Ask me anything about Leave, WFH, Performance, Compensation, POSH & more</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s)", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">📄 <b>{s["file"]}</b> — Page {s["page"]}<br>'
                        f'<i>{s["snippet"]}</i></div>', unsafe_allow_html=True)

prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask an HR question…") or prefill

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies…"):
            result = ask(user_input, retriever, chain)
        if result["oos"]:
            st.markdown(f'<div class="oos-card">⚠️ {result["answer"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(f"📎 {len(result['sources'])} source(s)", expanded=False):
                    for s in result["sources"]:
                        st.markdown(
                            f'<div class="source-card">📄 <b>{s["file"]}</b> — Page {s["page"]}<br>'
                            f'<i>{s["snippet"]}</i></div>', unsafe_allow_html=True)

    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })

st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#888;font-size:.8rem'>"
    "Zyro Dynamics HR Help Desk · RAG + Groq · Built with LangChain & Streamlit"
    "</div>", unsafe_allow_html=True)
