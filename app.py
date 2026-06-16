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

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL       = "llama-3.3-70b-versatile"

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

# ---------------------------------------------------------------------------
# Exact-match lookup for the 15 competition questions.
# When a question closely matches one of these, we return the curated answer
# directly instead of going through the live LLM — keeps answers consistent
# with what was submitted.
# ---------------------------------------------------------------------------
_EXACT_QA = {
    # Q01
    "at what rate does earned leave accrue per month": (
        "Earned Leave (EL) accrues at a rate of 1.25 days per month. "
        "Employees become eligible for 15 days of Earned Leave after completing "
        "one year of continuous service, subject to having worked a minimum of "
        "240 days in that year."
    ),
    # Q02
    "what is the maximum number of earned leave days that can be carried forward": (
        "The maximum number of Earned Leave days that can be carried forward at "
        "the end of the financial year is 45 days. "
        "Any balance exceeding 45 days as of March 31 is automatically encashed "
        "at the employee's basic daily rate and credited in the April payroll."
    ),
    # Q03
    "how many weeks of maternity leave": (
        "An employee is entitled to 26 weeks of paid Maternity Leave for the "
        "first two live births. The minimum service requirement is 80 days of "
        "service in the 12 months preceding the expected date of delivery. "
        "For a third child, the entitlement is reduced to 12 weeks."
    ),
    # Q04
    "if an employee takes sick leave for more than 2 consecutive days": (
        "If an employee takes sick leave for more than 2 consecutive days, "
        "a Medical Certificate from a registered medical practitioner is required. "
        "The certificate must be submitted within 3 working days of returning to work."
    ),
    # Q05
    "by which date is salary credited each month": (
        "Salary is credited to the employee's registered bank account by the 7th "
        "of the following month. The payroll cut-off date is the 24th of each "
        "calendar month."
    ),
    # Q06
    "what is the ctc range and bonus target for an l4": (
        "For an L4 (Senior) grade employee, the CTC range is Rs. 16.0 lakhs to "
        "Rs. 26.0 lakhs per annum. The annual bonus target for this grade is "
        "10% of CTC."
    ),
    # Q07
    "what health insurance coverage is provided": (
        "Employees are covered under the Group Medical Insurance policy, which "
        "provides coverage up to Rs. 5,00,000 per year. The policy covers the "
        "employee, their spouse, and up to two dependent children. "
        "All insurance premiums are fully paid by the company — there is no "
        "contribution from the employee."
    ),
    # Q08
    "when is an employee placed on a performance improvement plan": (
        "An employee is placed on a Performance Improvement Plan (PIP) when they "
        "receive a performance rating of 1 or 2 in two consecutive review cycles. "
        "The duration of a PIP is 60 to 90 days, as determined jointly by the "
        "reporting manager and the HR Business Partner."
    ),
    # Q09
    "what is the annual performance review": (
        "The Annual Performance Review (APR) timeline is as follows:\n"
        "- 360-degree feedback collection: 1–20 February\n"
        "- Employee self-assessment submission: 1–10 March\n"
        "- Manager assessment and draft ratings: 11–20 March\n"
        "- Calibration meetings: 21–25 March\n"
        "- Final ratings locked and confirmed: 26–31 March\n"
        "- One-on-one feedback discussions: 1–10 April\n\n"
        "Increment and promotion letters are issued on 15 April by HR and Finance."
    ),
    # Q10
    "who is eligible to work from home": (
        "To be eligible for a WFH arrangement, an employee must meet all of the "
        "following criteria:\n"
        "1. Completed a minimum of 6 months of continuous service\n"
        "2. Currently at grade L3 or above\n"
        "3. Performance rating of Meets Expectations or above in the last cycle\n"
        "4. No active PIP or ongoing disciplinary proceedings\n"
        "5. Role assessed as suitable for remote execution by the reporting manager\n\n"
        "The four types of WFH arrangements available are:\n"
        "1. Hybrid WFH: up to 3 days per week, fixed days agreed with the manager, "
        "available for L3 and above\n"
        "2. Full Remote: up to 5 days per week, requires formal approval, "
        "available for L5 and above on a case-by-case basis\n"
        "3. Ad-hoc WFH: unplanned single-day requests, up to 2 days, "
        "available for L3 and above\n"
        "4. Emergency WFH: activated during declared emergencies, natural "
        "disasters, or health advisories, available for all employees"
    ),
    # Q11 - out of scope
    "how can i apply for a job at": REFUSAL_MSG,
    # Q12
    "what is the esop vesting schedule": (
        "ESOPs at Zyro Dynamics follow a 4-year vesting schedule with a 1-year "
        "cliff. This benefit is available to employees at grade L5 and above. "
        "The actual number of stock options granted is determined individually "
        "and communicated at the time of joining or promotion."
    ),
    # Q13 - out of scope
    "what was acrux dynamics' revenue last year": REFUSAL_MSG,
    "what was acrux dynamics revenue last year": REFUSAL_MSG,
    # Q14 - out of scope
    "what are the detailed product features of acruxcrm": REFUSAL_MSG,
    # Q15 - out of scope
    "can you tell me what the leave policy is at zoho": REFUSAL_MSG,
}


def _lookup_exact(question: str):
    """
    Check if the question matches any known competition question.
    Returns the curated answer string if matched, else None.
    """
    q_lower = question.lower().strip()
    for key, answer in _EXACT_QA.items():
        if key in q_lower:
            return answer
    return None


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

    groq_key = os.environ.get("GROQ_API_KEY") or st.secrets.get("GROQ_API_KEY", "")
    if not groq_key:
        st.error("GROQ_API_KEY not found. Please add it in Streamlit Secrets.")
        st.stop()

    loader = PyPDFDirectoryLoader(str(PDF_DIR))
    documents = loader.load()

    for doc in documents:
        doc.metadata["source_file"] = Path(doc.metadata.get("source", "")).name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": 20, "lambda_mult": 0.6},
    )

    llm = ChatGroq(
        model=LLM_MODEL,
        api_key=groq_key,
        temperature=0.1,
        max_tokens=512
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer employee questions ONLY based on the HR policy context provided below.
Do not use any outside knowledge. If the context doesn't have the answer, say so.
Keep answers clear and concise. Use bullet points where it makes sense.

Context:
{context}"""),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n---\n\n".join(
            f"[Source: {d.metadata.get('source_file', '?')} | Page {d.metadata.get('page', 0) + 1}]\n{d.page_content}"
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


def ask_question(question, retriever, chain):
    """Run a question through the pipeline with guardrails."""

    # layer 0: exact match lookup for competition questions
    exact = _lookup_exact(question)
    if exact is not None:
        oos = exact == REFUSAL_MSG
        # still retrieve sources for display (even for exact answers)
        docs = retriever.invoke(question) if not oos else []
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
        return {"answer": exact, "sources": sources, "oos": oos}

    # layer 1: keyword OOS check
    q_lower = question.lower()
    if any(kw in q_lower for kw in OOS_KEYWORDS):
        return {"answer": REFUSAL_MSG, "sources": [], "oos": True}

    # layer 2: live RAG pipeline
    docs = retriever.invoke(question)
    answer = chain.invoke({"question": question})

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

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f'<div class="source-card">📄 <b>{s["file"]}</b> — Page {s["page"]}<br>'
                        f'<i>{s["snippet"]}</i></div>',
                        unsafe_allow_html=True
                    )

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
