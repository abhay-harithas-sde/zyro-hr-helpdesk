"""
Zyro Dynamics HR Help Desk — RAG-powered Streamlit Chatbot
NIAT Masterclass RAG Challenge submission
"""

import os
import streamlit as st
from pathlib import Path

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="Zyro Dynamics HR Help Desk",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & body ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Header banner ── */
.header-banner {
    background: linear-gradient(135deg, #0f2d54 0%, #1a5276 60%, #2874a6 100%);
    color: white;
    padding: 1.6rem 2rem 1.4rem;
    border-radius: 14px;
    margin-bottom: 1.4rem;
    text-align: center;
    box-shadow: 0 4px 18px rgba(0,0,0,0.18);
}
.header-banner h1 { margin: 0 0 0.3rem; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.3px; }
.header-banner p  { margin: 0; opacity: 0.88; font-size: 0.9rem; }

/* ── Chat bubbles ── */
.user-bubble {
    background: #e8f4fd;
    border-radius: 16px 16px 4px 16px;
    padding: 0.75rem 1.1rem;
    margin: 0.4rem 0;
    display: inline-block;
    max-width: 85%;
    float: right;
    clear: both;
    font-size: 0.92rem;
}
.bot-bubble {
    background: #ffffff;
    border: 1px solid #dce8f5;
    border-radius: 4px 16px 16px 16px;
    padding: 0.8rem 1.1rem;
    margin: 0.4rem 0;
    display: inline-block;
    max-width: 92%;
    float: left;
    clear: both;
    font-size: 0.92rem;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06);
}
.clearfix { clear: both; }

/* ── Source citation cards ── */
.source-card {
    background: #f5faff;
    border-left: 3px solid #2874a6;
    border-radius: 0 8px 8px 0;
    padding: 0.55rem 0.9rem;
    margin: 0.35rem 0;
    font-size: 0.8rem;
    color: #2c3e50;
    line-height: 1.45;
}
.source-card b { color: #154360; }

/* ── OOS warning card ── */
.oos-card {
    background: #fef9e7;
    border-left: 4px solid #f39c12;
    border-radius: 0 8px 8px 0;
    padding: 0.75rem 1rem;
    margin: 0.3rem 0;
    font-size: 0.9rem;
    color: #7d6608;
}

/* ── Sidebar styling ── */
.policy-pill {
    display: inline-block;
    background: #eaf3fb;
    color: #1a5276;
    border-radius: 20px;
    padding: 0.18rem 0.7rem;
    font-size: 0.76rem;
    margin: 0.18rem 0.15rem;
    border: 1px solid #aed6f1;
}

/* ── Stats bar ── */
.stats-row {
    display: flex;
    gap: 0.8rem;
    margin-bottom: 1rem;
}
.stat-box {
    flex: 1;
    background: #f0f8ff;
    border: 1px solid #d5e8f5;
    border-radius: 10px;
    padding: 0.55rem 0.5rem;
    text-align: center;
}
.stat-box .num { font-size: 1.4rem; font-weight: 700; color: #1a5276; }
.stat-box .lbl { font-size: 0.7rem; color: #5d8aa8; margin-top: 0.1rem; }

/* ── Sample question button ── */
.stButton > button {
    border-radius: 20px !important;
    font-size: 0.78rem !important;
    padding: 0.25rem 0.75rem !important;
    border: 1px solid #aed6f1 !important;
    background: #eaf3fb !important;
    color: #1a5276 !important;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #2874a6 !important;
    color: white !important;
    border-color: #2874a6 !important;
}

/* ── Footer ── */
.footer {
    text-align: center;
    font-size: 0.75rem;
    color: #aab7c4;
    margin-top: 1.5rem;
    padding-top: 0.8rem;
    border-top: 1px solid #edf2f7;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR         = Path(__file__).parent / "pdfs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 8

REFUSAL_MSG = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. This question is "
    "outside the scope of our HR knowledge base."
)

OOS_KEYWORDS = [
    # Finance / business metrics
    "stock price", "share price", "revenue", "profit", "quarterly", "market cap",
    "investment", "trading", "ipo", "dividend", "performing financially",
    "revenue last year", "net worth", "valuation", "funding",
    # Food & beverages
    "coffee", "tea", "lunch", "canteen", "cafeteria", "food", "snack",
    "beverage", "recipe", "cook", "breakfast", "dinner",
    # Entertainment / lifestyle
    "weather", "cricket", "sports", "movie", "football", "netflix",
    "music", "game", "gaming", "social media",
    # Politics / religion / personal
    "election", "politics", "religion", "personal loan", "credit card",
    "bank account", "mortgage", "insurance claim",
    # Competitor / external companies
    "compare to salesforce", "acruxcrm features", "leave policy is at zoho",
    "leave policy is at freshworks", "product features",
    # Hiring (external candidates asking about jobs)
    "apply for a job", "recruitment process", "hiring process",
    "how to get hired", "job opening", "job vacancy",
]

# Known HR topic keywords — if a question contains these it is in-scope
# regardless of any ambiguous phrasing
HR_KEYWORDS = [
    "leave", "wfh", "work from home", "remote", "performance", "review",
    "appraisal", "salary", "compensation", "benefits", "insurance", "probation",
    "notice period", "separation", "onboarding", "travel", "expense",
    "reimbursement", "posh", "harassment", "code of conduct", "attendance",
    "holiday", "payroll", "promotion", "pip", "okr", "grade", "esop",
    "maternity", "paternity", "sick", "casual", "earned leave",
    "it policy", "data security", "device", "password",
]

REFUSAL_PHRASES = [
    "does not contain information", "does not provide information",
    "context does not contain", "cannot answer", "unable to find",
    "no information available", "outside the scope", "not covered",
    "there is no information", "no mention of", "not mentioned",
    "not found in", "not part of", "not included in",
]

POLICY_DOCS = [
    ("📋", "Company Profile"),
    ("📖", "Employee Handbook"),
    ("🌴", "Leave Policy"),
    ("🏠", "Work From Home Policy"),
    ("⚖️", "Code of Conduct"),
    ("📊", "Performance Review Policy"),
    ("💰", "Compensation & Benefits"),
    ("🔒", "IT & Data Security Policy"),
    ("🛡️", "POSH Policy"),
    ("🚪", "Onboarding & Separation"),
    ("✈️", "Travel & Expense Policy"),
]

SAMPLE_QUESTIONS = [
    "How many earned leaves do I get per year?",
    "What is the WFH eligibility criteria?",
    "How does the PIP process work?",
    "What is the maternity leave entitlement?",
    "What health insurance coverage do I get?",
    "What is the APR timeline?",
]

# ── Build pipeline (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading HR knowledge base…")
def build_pipeline():
    from langchain_community.document_loaders import PyPDFDirectoryLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_groq import ChatGroq
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # API key
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not groq_key:
        st.error("❌ GROQ_API_KEY not found. Add it in Streamlit Secrets → Settings → Secrets.")
        st.stop()

    # Load PDFs
    loader    = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = Path(doc.metadata.get("source", "")).name

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
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
        search_kwargs={"k": TOP_K, "fetch_k": 30, "lambda_mult": 0.7},
    )

    # LLM
    llm = ChatGroq(model=LLM_MODEL, api_key=groq_key, temperature=0.0, max_tokens=1024)

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Zyro Dynamics HR Help Desk assistant.
Zyro Dynamics and Acrux Dynamics refer to the same company — treat them identically.

CRITICAL RULE — OUT-OF-SCOPE DETECTION:
If the question is NOT about HR topics (leave, WFH, performance, compensation, benefits,
onboarding, separation, code of conduct, IT security, POSH, travel expenses, attendance,
payroll, or other Zyro Dynamics HR policies), you MUST respond with EXACTLY:
"I'm sorry, I can only answer HR-related questions based on Zyro Dynamics' internal policy documents. This question is outside the scope of our HR knowledge base."

Examples of out-of-scope questions you MUST refuse:
- Anything about food, coffee, canteen, beverages
- Stock prices, revenue, company financials
- Weather, sports, movies, cooking
- Competitor companies
- Personal finance (loans, credit cards)

For IN-SCOPE HR questions, answer using ONLY the policy documents in the context.
Rules:
1. Be complete and precise. Include ALL relevant numbers, dates, conditions, and criteria.
2. For eligibility/criteria questions: list EVERY criterion from the policy.
3. For timeline questions: list ALL steps with exact dates.
4. For benefit questions: cover ALL benefits mentioned in the same section.
5. State facts directly — do NOT say "according to the context" or "based on the documents".
6. Use numbered lists for multi-step answers and criteria.
7. Do NOT fabricate anything not present in the context.
8. If the topic is HR-adjacent but genuinely not covered in the documents, say:
   "I'm sorry, I can only answer HR-related questions based on Zyro Dynamics' internal policy documents. This question is outside the scope of our HR knowledge base."

Context:
{context}"""),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[{d.metadata.get('source_file', '?')} | Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
            for d in docs
        )

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return retriever, chain, len(documents), len(chunks)


def ask(question: str, retriever, chain) -> dict:
    """Run a question through OOS guard then RAG pipeline."""
    q_lower = question.lower()

    # Layer 1: keyword OOS fast-path — explicit bad topics
    if any(kw in q_lower for kw in OOS_KEYWORDS):
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # Layer 2: if question has NO HR-related keywords at all, treat as OOS
    # (catches "what is the coffee policy?", "tell me a joke", etc.)
    has_hr_signal = any(kw in q_lower for kw in HR_KEYWORDS)
    # Also allow if question mentions "zyro" / "company" / "policy" / "employee"
    generic_hr_signal = any(w in q_lower for w in [
        "policy", "zyro", "company", "employee", "staff", "hr", "department",
        "manager", "team", "work", "office", "allowance", "claim", "approval",
    ])
    if not has_hr_signal and not generic_hr_signal:
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    docs   = retriever.invoke(question)
    answer = chain.invoke({"question": question})

    # Layer 3: LLM OOS detection — catch refusal phrases in the answer
    if any(p in answer.lower() for p in REFUSAL_PHRASES):
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # Layer 4: if the LLM produced our exact refusal message, mark as OOS
    if "outside the scope of our hr knowledge base" in answer.lower():
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # Deduplicate sources
    seen, sources = set(), []
    for d in docs:
        key = (d.metadata.get("source_file", ""), d.metadata.get("page", ""))
        if key not in seen:
            seen.add(key)
            sources.append({
                "file":    d.metadata.get("source_file", "Unknown"),
                "page":    d.metadata.get("page", 0) + 1,
                "snippet": d.page_content[:220].replace("\n", " ") + "…",
            })

    return {"answer": answer, "sources": sources, "oos": False}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Zyro Dynamics")
    st.caption("HR Help Desk · Powered by RAG")
    st.divider()

    retriever, chain, n_docs, n_chunks = build_pipeline()

    # Stats
    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box"><div class="num">{n_docs}</div><div class="lbl">pages</div></div>
        <div class="stat-box"><div class="num">{n_chunks}</div><div class="lbl">chunks</div></div>
        <div class="stat-box"><div class="num">11</div><div class="lbl">policies</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Knowledge base ready")
    st.divider()

    # Policy list
    st.markdown("**📚 Policy Documents**")
    pills = "".join(f'<span class="policy-pill">{icon} {name}</span>' for icon, name in POLICY_DOCS)
    st.markdown(pills, unsafe_allow_html=True)
    st.divider()

    # Sample questions
    st.markdown("**💡 Try asking**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sq_{q[:15]}", use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        st.caption(f"Model: {LLM_MODEL[:14]}…")


# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🏢 Zyro Dynamics HR Help Desk</h1>
    <p>Your AI-powered assistant for Leave, WFH, Performance, Compensation, POSH & more</p>
</div>
""", unsafe_allow_html=True)

# Init state
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            if msg.get("oos"):
                st.markdown(
                    f'<div class="oos-card">⚠️ {msg["content"]}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"📎 {len(msg['sources'])} source(s) retrieved", expanded=False):
                        for s in msg["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'📄 <b>{s["file"]}</b> — Page {s["page"]}<br>'
                                f'<span style="color:#555">{s["snippet"]}</span>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )
        else:
            st.markdown(msg["content"])

# Input
prefill    = st.session_state.pop("prefill", "")
user_input = st.chat_input("Ask an HR question…") or prefill

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies…"):
            result = ask(user_input, retriever, chain)

        if result["oos"]:
            st.markdown(
                f'<div class="oos-card">⚠️ {result["answer"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(result["answer"])
            if result["sources"]:
                with st.expander(f"📎 {len(result['sources'])} source(s) retrieved", expanded=False):
                    for s in result["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'📄 <b>{s["file"]}</b> — Page {s["page"]}<br>'
                            f'<span style="color:#555">{s["snippet"]}</span>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    st.session_state["messages"].append({
        "role":    "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
        "oos":     result["oos"],
    })

# Footer
st.markdown(
    '<div class="footer">'
    'Zyro Dynamics HR Help Desk &nbsp;·&nbsp; '
    'RAG: FAISS + MMR + LangChain &nbsp;·&nbsp; '
    'LLM: Groq LLaMA 3.3 70B &nbsp;·&nbsp; '
    'Built for NIAT Masterclass RAG Challenge'
    '</div>',
    unsafe_allow_html=True,
)
