# Zyro Dynamics HR Help Desk Chatbot

This is my submission for the NIAT Masterclass RAG Challenge on Kaggle. The idea was to build a chatbot that can answer HR-related questions from company policy documents without having to manually search through PDFs every time.

---

## What I built

A RAG (Retrieval-Augmented Generation) pipeline that:
- Loads 11 HR policy PDFs from Zyro Dynamics
- Converts them into searchable chunks using embeddings
- Retrieves relevant policy text when a question is asked
- Passes that text to an LLM (Groq) to generate a grounded answer
- Refuses to answer anything not related to HR policies

The chatbot is deployed as a Streamlit web app so anyone can use it from a browser.

---

## How RAG works (in simple terms)

Normal LLMs just answer from their training data which can be wrong or outdated. RAG fixes this by:

1. First searching a database of your actual documents for relevant text
2. Then feeding that text to the LLM and saying "answer only from this"

So the answers are always based on the real company policies, not made-up stuff.

---

## The 11 HR documents used

- Company Profile
- Employee Handbook
- Leave Policy (EL, SL, Maternity, Paternity)
- Work From Home Policy
- Code of Conduct
- Performance Review Policy
- Compensation and Benefits Policy
- IT and Data Security Policy
- POSH Policy
- Onboarding and Separation Policy
- Travel and Expense Policy

---

## How the pipeline works

**Step 1 - Load PDFs**  
Used `PyPDFDirectoryLoader` to read all 11 PDFs. Each page becomes a document with metadata like the filename and page number.

**Step 2 - Split into chunks**  
Each document is split into smaller pieces (800 characters with 150 overlap) so the retriever can find specific sections instead of entire pages.

**Step 3 - Create embeddings**  
Used `sentence-transformers/all-MiniLM-L6-v2` to convert each chunk into a vector. Similar text ends up with similar vectors.

**Step 4 - FAISS index**  
All vectors are stored in FAISS (a fast vector search library by Facebook). When a question comes in, FAISS finds the closest matching chunks.

**Step 5 - MMR retrieval**  
Instead of just grabbing the top 5 most similar chunks, I used MMR (Maximal Marginal Relevance) which also considers diversity. This avoids getting 5 nearly identical chunks from the same paragraph.

**Step 6 - Groq LLM**  
The retrieved chunks are passed to `llama-3.3-70b-versatile` via Groq API with a prompt that says to only answer from the given context.

**Step 7 - Guardrails**  
Two checks to catch out-of-scope questions:
- A keyword list that immediately blocks things like "stock price", "cricket score", "recipe" etc.
- A check on the LLM's response — if it says "the context does not contain this", the answer is replaced with a polite refusal

---

## How the submission CSV works

The competition encrypts both the questions and answers using Fernet symmetric encryption. The secret key is provided in the starter notebook.

```
Encrypted question in CSV
        ↓
Decrypt with SUBMISSION_SECRET key
        ↓
Plain text question → run through RAG bot → plain text answer
        ↓
Encrypt the answer with same key
        ↓
answer_enc column in submission.csv
```

The evaluators have the same key, so they decrypt your answers and compare them against the ground truth using semantic similarity scoring. This means you can't hardcode answers — they have to actually make sense semantically.

Each row in submission.csv has:
- `question_id` — Q01 to Q15
- `question_enc` — encrypted version of the question
- `answer_enc` — encrypted version of your answer
- `streamlit_link` — your deployed chatbot URL
- `langsmith_link` — your LangSmith trace URL

---

## Files

```
run_pipeline.py        - runs the full RAG pipeline locally and saves answers
fix_and_submit.py      - applies post-processing and generates submission.csv
get_trace_url.py       - sends runs to LangSmith and fetches the trace URL
kaggle_submit.py       - submits submission.csv to Kaggle via API
answers_preview.json   - the 15 answers in readable form (for review)
submission.csv         - final file uploaded to Kaggle

streamlit-deploy/
    app.py             - the chatbot web app
    requirements.txt   - dependencies for Streamlit Cloud
    pdfs/              - all 11 HR policy PDFs bundled for deployment
```

---

## Tech used

| What | Tool |
|------|------|
| LLM | Groq — llama-3.3-70b-versatile |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector store | FAISS |
| RAG framework | LangChain (LCEL) |
| Tracing | LangSmith |
| Frontend | Streamlit |
| Deployment | Streamlit Community Cloud |

---

## Setup

```bash
pip install langchain langchain-community langchain-groq langchain-huggingface \
    langsmith faiss-cpu pypdf sentence-transformers streamlit python-dotenv cryptography
```

Create a `.env` file:
```
GROQ_API_KEY=your_key
LANGCHAIN_API_KEY=your_key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=zyro-rag-challenge
```

Run locally:
```bash
python run_pipeline.py
streamlit run streamlit-deploy/app.py
```

---

## Live links

- Streamlit app: https://zyro-hr-appdesk-4wmdse8pom22pxvodvzb4w.streamlit.app/
- LangSmith trace: https://smith.langchain.com/public/9e3d2fda-0fec-4640-b410-4eff8b846522/r

---

## Scoring breakdown

Q01-Q10 are in-scope HR questions worth 8 points each (80 total).  
Q11-Q15 are out-of-scope questions — you get points for correctly refusing to answer them (4 pts each, 20 total).

Total possible: 100 points.

---

## What I learned

- RAG is a much better approach than just prompting an LLM directly when you need answers to be accurate and based on specific documents
- Chunk size and overlap matter a lot — too small and you lose context, too large and retrieval gets noisy
- MMR retrieval is noticeably better than plain similarity search for policy documents where multiple sections say similar things
- LangSmith makes it really easy to debug what the retriever is actually fetching for each question
