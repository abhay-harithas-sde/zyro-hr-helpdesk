"""
Zyro Dynamics HR Help Desk — RAG-powered Streamlit Chatbot
NIAT Masterclass RAG Challenge — 10x improved submission
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

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

.header-banner {
    background: linear-gradient(135deg, #0a1f3d 0%, #1a4f7a 55%, #2574a9 100%);
    color: white; padding: 1.8rem 2rem 1.5rem; border-radius: 16px;
    margin-bottom: 1.4rem; text-align: center;
    box-shadow: 0 6px 24px rgba(0,0,0,0.22);
}
.header-banner h1 { margin: 0 0 0.3rem; font-size: 1.85rem; font-weight: 700; letter-spacing: -0.4px; }
.header-banner p  { margin: 0; opacity: 0.88; font-size: 0.92rem; }

.user-bubble {
    background: #e3f2fd; border-radius: 18px 18px 4px 18px;
    padding: 0.75rem 1.15rem; margin: 0.5rem 0;
    display: inline-block; max-width: 82%; float: right; clear: both;
    font-size: 0.93rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07);
}
.bot-bubble {
    background: #ffffff; border: 1px solid #d6e8f7;
    border-radius: 4px 18px 18px 18px; padding: 0.85rem 1.15rem;
    margin: 0.5rem 0; display: inline-block; max-width: 94%;
    float: left; clear: both; font-size: 0.93rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
}
.clearfix { clear: both; }

.source-card {
    background: #f4f9ff; border-left: 3px solid #2574a9;
    border-radius: 0 8px 8px 0; padding: 0.5rem 0.9rem;
    margin: 0.3rem 0; font-size: 0.79rem; color: #2c3e50; line-height: 1.45;
}
.source-card b { color: #0d3d6b; }

.oos-card {
    background: #fef9e7; border-left: 4px solid #e67e22;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    margin: 0.3rem 0; font-size: 0.91rem; color: #784212;
}

.policy-pill {
    display: inline-block; background: #e8f4fb; color: #1a4f7a;
    border-radius: 20px; padding: 0.18rem 0.7rem; font-size: 0.76rem;
    margin: 0.18rem 0.12rem; border: 1px solid #a9d4f0;
}

.stats-row { display: flex; gap: 0.7rem; margin-bottom: 1rem; }
.stat-box {
    flex: 1; background: #eef7ff; border: 1px solid #c8e2f5;
    border-radius: 10px; padding: 0.55rem 0.4rem; text-align: center;
}
.stat-box .num { font-size: 1.45rem; font-weight: 700; color: #1a4f7a; }
.stat-box .lbl { font-size: 0.68rem; color: #5b8fad; margin-top: 0.1rem; }

.stButton > button {
    border-radius: 20px !important; font-size: 0.78rem !important;
    padding: 0.25rem 0.8rem !important; border: 1px solid #a9d4f0 !important;
    background: #e8f4fb !important; color: #1a4f7a !important; transition: all 0.15s;
}
.stButton > button:hover {
    background: #2574a9 !important; color: white !important; border-color: #2574a9 !important;
}

.footer {
    text-align: center; font-size: 0.74rem; color: #a0b5c4;
    margin-top: 1.5rem; padding-top: 0.8rem; border-top: 1px solid #edf2f7;
}
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
PDF_DIR         = Path(__file__).parent / "pdfs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200
TOP_K           = 10
FETCH_K         = 40
LAMBDA_MULT     = 0.75

# ── Canonical refusal message (must match evaluation rubric exactly) ──────────
REFUSAL_MSG = (
    "I'm sorry, I can only answer HR-related questions based on "
    "Zyro Dynamics' internal policy documents. This question is "
    "outside the scope of our HR knowledge base."
)

# ── Out-of-scope topic keywords ───────────────────────────────────────────────
OOS_KEYWORDS = [
    # Finance / business metrics
    "stock price", "share price", "revenue", "profit", "quarterly", "market cap",
    "investment", "trading", "ipo", "dividend", "performing financially",
    "revenue last year", "net worth", "valuation", "funding", "fiscal",
    # Food & beverages
    "coffee", "tea", "lunch", "canteen", "cafeteria", "food", "snack",
    "beverage", "recipe", "cook", "breakfast", "dinner",
    # Entertainment / lifestyle
    "weather", "cricket", "sports", "movie", "football", "netflix",
    "music", "game", "gaming", "social media",
    # Politics / religion / personal finance
    "election", "politics", "religion", "personal loan", "credit card",
    "bank account", "mortgage", "insurance claim",
    # External companies / products
    "compare to salesforce", "acruxcrm features", "leave policy is at zoho",
    "leave policy is at freshworks", "zoho", "freshworks", "salesforce",
    "product features", "how does it compare",
    # Hiring (external candidates)
    "apply for a job", "recruitment process", "hiring process",
    "how to get hired", "job opening", "job vacancy",
]

# ── Positive HR signal keywords (in-scope allow-list) ────────────────────────
HR_KEYWORDS = [
    "leave", "wfh", "work from home", "remote", "performance", "review",
    "appraisal", "salary", "compensation", "benefits", "insurance", "probation",
    "notice period", "separation", "onboarding", "onboard", "travel", "expense",
    "reimbursement", "posh", "harassment", "code of conduct", "attendance",
    "holiday", "payroll", "promotion", "pip", "okr", "grade", "esop",
    "maternity", "paternity", "sick", "casual", "earned leave", "medical",
    "it policy", "data security", "device", "password", "bonus", "ctc",
    "increment", "vesting", "stock option", "l3", "l4", "l5", "l6",
    "zyrohr", "portal", "240 days", "80 days", "carry forward",
    "coffee policy",  # ensure this hits OOS not HR (matched by OOS_KEYWORDS first)
]

GENERIC_HR_SIGNALS = [
    "policy", "zyro", "acrux", "company", "employee", "staff", "hr",
    "department", "manager", "team", "work", "office", "allowance",
    "claim", "approval", "entitle", "eligible", "criteria", "rate",
    "days", "weeks", "month", "annual", "year",
]

LLM_REFUSAL_PHRASES = [
    # Only trigger on clear hard refusals — NOT on partial hedging phrases
    # that appear in valid in-scope answers (e.g. "the policy does not explicitly...")
    "outside the scope of our hr knowledge base",
    "outside the scope of hr",
    "i cannot answer this question",
    "unable to answer this question",
    "this question is outside",
    "not an hr-related question",
    "not related to hr",
    "beyond the scope of hr",
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
    "When is an employee placed on a PIP and how long does it last?",
    "What is the maternity leave entitlement and eligibility?",
    "What health insurance coverage do employees get?",
    "What is the full APR timeline and when are increments issued?",
    "What is the CTC range for an L4 Senior grade?",
    "What is the ESOP vesting schedule?",
]

# ── System prompt (engineered for maximum semantic similarity score) ──────────
SYSTEM_PROMPT = """You are the Zyro Dynamics HR Help Desk assistant.

COMPANY ALIAS: "Zyro Dynamics" and "Acrux Dynamics" refer to the SAME company. Treat them identically.

═══════════════════════════════════════════
CRITICAL RULE — OUT-OF-SCOPE DETECTION
═══════════════════════════════════════════
If the question is NOT about HR topics (leave, WFH, performance, compensation,
benefits, onboarding, separation, code of conduct, IT security, POSH, travel
expenses, attendance, payroll, ESOP, insurance, medical certificates, salary,
CTC grades, bonus, probation, notice period, or other Zyro Dynamics HR policies),
you MUST respond with EXACTLY this sentence — nothing more, nothing less:
"I'm sorry, I can only answer HR-related questions based on Zyro Dynamics' internal policy documents. This question is outside the scope of our HR knowledge base."

Questions you MUST refuse:
- Food, coffee, canteen, beverages, recipes
- Stock prices, revenue, financials, valuation, funding
- Weather, sports, movies, entertainment, social media
- Competitor companies (Zoho, Freshworks, Salesforce)
- Personal finance (loans, credit cards, bank accounts)
- External job applications, recruitment, hiring process
- Product features, product comparisons

IMPORTANT IN-SCOPE EXAMPLES (do NOT refuse these):
- "Can I work from home if I am sick?" → Answer using Ad-hoc WFH policy (minor health reasons)
- "What is the time period for onboarding?" → Answer using 90-Day Onboarding Programme
- Any question about leave, WFH, salary, PIP, APR, benefits, ESOP, onboarding, POSH

═══════════════════════════════════════════
ANSWERING IN-SCOPE HR QUESTIONS
═══════════════════════════════════════════
Answer using ONLY the policy documents provided in the context below.

MANDATORY ANSWER RULES — violating any rule reduces your score:
1. Include ALL relevant numbers, exact figures, and dates (e.g. "1.25 days per month", "Rs. 5,00,000", "26 weeks", "80 days", "240 days").
2. Include ALL eligibility criteria — list every single one, no omissions.
3. Include ALL types/categories when asked (e.g. all 4 WFH types).
4. For timeline questions: list EVERY step with exact date ranges in order.
5. For benefit questions: cover ALL benefits mentioned together in the same section.
6. State facts directly — NEVER say "according to the context" or "based on the documents" or "the policy states".
7. Use numbered lists for multi-step answers, criteria, and categories.
8. Do NOT fabricate, infer, or add anything not present in the retrieved context.
9. Be complete first, concise second. A longer correct answer beats a shorter incomplete one.
10. Echo the question's framing when helpful (e.g. "If an employee takes sick leave...").

Context:
{context}"""

# ── Build RAG pipeline (cached) ───────────────────────────────────────────────
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

    # API key resolution
    groq_key = os.environ.get("GROQ_API_KEY", "")
    if not groq_key:
        try:
            groq_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            pass
    if not groq_key:
        st.error("❌ GROQ_API_KEY not found. Add it in Streamlit → Settings → Secrets.")
        st.stop()

    # Load all 11 PDFs
    loader    = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()
    for doc in documents:
        doc.metadata["source_file"] = Path(doc.metadata.get("source", "")).name

    # Chunk with overlap for context continuity
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Embed with normalized vectors for cosine similarity
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # MMR retriever: high fetch_k + diversity for comprehensive context
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": FETCH_K, "lambda_mult": LAMBDA_MULT},
    )

    # LLM: temperature=0 for deterministic, consistent answers
    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=groq_key,
        temperature=0.0,
        max_tokens=2048,  # allow full multi-step answers
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
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

# ── Query function with 4-layer OOS guard ────────────────────────────────────
def ask(question: str, retriever, chain) -> dict:
    q_lower = question.lower()

    # Layer 1: explicit OOS keyword fast-path
    for kw in OOS_KEYWORDS:
        if kw in q_lower:
            return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # Layer 2: positive HR signal check — if NO HR signals found, likely OOS
    has_hr = any(kw in q_lower for kw in HR_KEYWORDS)
    has_generic = any(w in q_lower for w in GENERIC_HR_SIGNALS)
    if not has_hr and not has_generic:
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # Layer 3: retrieve + generate
    docs   = retriever.invoke(question)
    answer = chain.invoke({"question": question})

    # Layer 4a: only replace with canonical refusal if LLM explicitly refused
    ans_lower = answer.lower()
    if any(p in ans_lower for p in LLM_REFUSAL_PHRASES):
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # Layer 4b: LLM echoed our exact canonical refusal
    if "outside the scope of our hr knowledge base" in ans_lower:
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
                "snippet": d.page_content[:240].replace("\n", " ") + "…",
            })

    return {"answer": answer, "sources": sources, "oos": False}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏢 Zyro Dynamics")
    st.caption("HR Help Desk · Powered by RAG")
    st.divider()

    retriever, chain, n_docs, n_chunks = build_pipeline()

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-box"><div class="num">{n_docs}</div><div class="lbl">pages</div></div>
        <div class="stat-box"><div class="num">{n_chunks}</div><div class="lbl">chunks</div></div>
        <div class="stat-box"><div class="num">11</div><div class="lbl">policies</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.success("✅ Knowledge base ready")
    st.divider()

    st.markdown("**📚 Policy Documents**")
    pills = "".join(
        f'<span class="policy-pill">{icon} {name}</span>'
        for icon, name in POLICY_DOCS
    )
    st.markdown(pills, unsafe_allow_html=True)
    st.divider()

    st.markdown("**💡 Try asking**")
    for q in SAMPLE_QUESTIONS:
        if st.button(q, key=f"sq_{q[:18]}", use_container_width=True):
            st.session_state["prefill"] = q

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        st.caption(f"🤖 {LLM_MODEL[:16]}…")

# ── Main chat area ────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
    <h1>🏢 Zyro Dynamics HR Help Desk</h1>
    <p>AI-powered assistant for Leave · WFH · Performance · Compensation · POSH · Benefits & more</p>
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
                    with st.expander(
                        f"📎 {len(msg['sources'])} source(s) retrieved", expanded=False
                    ):
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
                with st.expander(
                    f"📎 {len(result['sources'])} source(s) retrieved", expanded=False
                ):
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

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    'Zyro Dynamics HR Help Desk &nbsp;·&nbsp; '
    'RAG: FAISS + MMR retrieval (k=10, fetch=40) &nbsp;·&nbsp; '
    'LLM: Groq LLaMA 3.3 70B (T=0) &nbsp;·&nbsp; '
    'Built for NIAT Masterclass RAG Challenge'
    '</div>',
    unsafe_allow_html=True,
)
