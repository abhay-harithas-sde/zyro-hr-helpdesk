# HR Help Desk chatbot for Zyro Dynamics
# Built for NIAT Masterclass RAG Challenge
# Using LangChain + FAISS + Groq + Streamlit

import os
import streamlit as st
from pathlib import Path

st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# some basic styling
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1e3a5f, #2d6a9f);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 12px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.main-header h1 { margin: 0; font-size: 1.8rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.85; font-size: 0.95rem; }

.source-card {
    background: #f0f7ff;
    border-left: 4px solid #2d6a9f;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
    font-size: 0.82rem;
    color: #333;
}
.oos-card {
    background: #fff3cd;
    border-left: 4px solid #ffc107;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    margin: 0.4rem 0;
}
</style>
""", unsafe_allow_html=True)

# PDFs are in the pdfs/ folder next to this file
PDF_DIR = Path(__file__).parent / "pdfs"

# chunking settings - tried a few values, 800/150 worked best for these docs
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150
TOP_K         = 5

# using free huggingface model for embeddings so no extra API key needed
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"  # llama3-8b was decommissioned, using this now

# keywords that clearly indicate out-of-scope questions
OOS_KEYWORDS = [
    "stock price", "share price", "revenue", "profit", "quarterly", "market cap",
    "investment", "trading", "ipo", "dividend", "competitor",
    "weather", "cricket", "sports", "movie", "recipe", "cook",
    "election", "politics", "religion", "personal loan", "credit card",
    "net worth", "apply for a job", "recruitment process", "hiring process",
    "product features", "compare to salesforce", "acruxcrm features",
    "leave policy is at zoho", "leave policy is at freshworks",
    "performing financially", "revenue last year",
]

REFUSAL_MSG = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. Your question appears to be "
    "outside the scope of our HR knowledge base."
)


# this builds the whole pipeline - cached so it only runs once per session
@st.cache_resource(show_spinner="Loading HR knowledge base, please wait...")
def build_pipeline():
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # get groq key from streamlit secrets or env
    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        st.error("GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
        st.stop()

    # step 1: load all PDFs from the pdfs folder
    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()

    # tag each page with its filename for citations later
    for doc in documents:
        doc.metadata["source_file"] = Path(doc.metadata.get("source", "")).name

    # step 2: split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    # step 3: create embeddings and build FAISS index
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # using MMR retrieval to get diverse results (not just the top similar ones)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.6},
    )

    # step 4: setup the LLM
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=groq_key,
        temperature=0.1,
        max_tokens=512
    )

    # system prompt - telling it to ONLY use the retrieved context
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions ONLY based on the HR policy context provided below.
Do not use any outside knowledge. If the context doesn't have the answer, say so.
Keep answers clear and concise. Use bullet points where it makes sense.

Context:
{context}"""),
        ("human", "{question}"),
    ])

    # helper to format retrieved docs with source info
    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_file', '?')} | Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
            for d in docs
        )

    # build the RAG chain using LCEL
    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain, len(documents), len(chunks)


def ask_question(question, retriever, chain):
    """run a question through the pipeline with guardrails"""

    # layer 1 guardrail: keyword check
    q_lower = question.lower()
    if any(kw in q_lower for kw in OOS_KEYWORDS):
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # get answer from RAG chain
    docs = retriever.invoke(question)
    answer = chain.invoke({"question": question})

    # layer 2 guardrail: if LLM says it can't find info, treat as out of scope
    refusal_phrases = [
        "does not contain information",
        "does not provide information",
        "context does not contain",
        "cannot answer",
        "no information available",
        "outside the scope",
    ]
    if any(phrase in answer.lower() for phrase in refusal_phrases):
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # collect unique sources for citation
    seen = set()
    sources = []
    for d in docs:
        key = (d.metadata.get("source_file", ""), d.metadata.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file": d.metadata.get("source_file", "Unknown"),
                "page": d.metadata.get("page", 0) + 1,
                "snippet": d.page_content[:200].replace("\n", " ") + "...",
            })

    return {"answer": answer, "sources": sources, "oos": False}


# sidebar
with st.sidebar:
    st.markdown("## 🏢 Zyro Dynamics")
    st.markdown("**HR Help Desk**")
    st.divider()

    # build pipeline when app loads
    retriever, chain, n_docs, n_chunks = build_pipeline()
    st.success("Knowledge base loaded!")
    st.markdown(f"- {n_docs} pages from 11 PDFs")
    st.markdown(f"- {n_chunks} text chunks indexed")
    st.divider()

    st.markdown("**Available Policies**")
    policies = [
        "Company Profile", "Employee Handbook", "Leave Policy",
        "Work From Home Policy", "Code of Conduct",
        "Performance Review", "Compensation & Benefits",
        "IT & Data Security", "POSH Policy",
        "Onboarding & Separation", "Travel & Expense",
    ]
    for p in policies:
        st.markdown(f"• {p}")

    st.divider()
    st.markdown("**Sample questions**")
    samples = [
        "How many earned leaves per year?",
        "What is the WFH policy?",
        "How does PIP work?",
        "What is maternity leave?",
    ]
    for s in samples:
        if st.button(s, key=s, use_container_width=True):
            st.session_state["prefill"] = s

    st.divider()
    if st.button("Clear Chat", use_container_width=True):
        st.session_state["messages"] = []
        st.rerun()


# main chat area
st.markdown("""
<div class="main-header">
  <h1>🏢 Zyro Dynamics HR Help Desk</h1>
  <p>Ask anything about Leave, WFH, Performance, Compensation, POSH and more</p>
</div>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# show chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        # show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">📄 <b>{s["file"]}</b> — Page {s["page"]}<br>'
                        f'<i>{s["snippet"]}</i></div>',
                        unsafe_allow_html=True
                    )

# handle new input
prefill = st.session_state.pop("prefill", "")
user_input = st.chat_input("Type your HR question here...") or prefill

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Looking through HR policies..."):
            result = ask_question(user_input, retriever, chain)

        if result["oos"]:
            st.markdown(
                f'<div class="oos-card">⚠️ {result["answer"]}</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(f"Sources ({len(result['sources'])})", expanded=False):
                    for s in result["sources"]:
                        st.markdown(
                            f'<div class="source-card">📄 <b>{s["file"]}</b> — Page {s["page"]}<br>'
                            f'<i>{s["snippet"]}</i></div>',
                            unsafe_allow_html=True
                        )

    st.session_state["messages"].append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })

st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.8rem;'>"
    "Built for NIAT Masterclass RAG Challenge | Powered by Groq + LangChain + Streamlit"
    "</div>",
    unsafe_allow_html=True
)
