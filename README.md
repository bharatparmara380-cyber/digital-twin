# 🤖 Digital Twin — Your Personal AI Agent

A smart AI version of yourself powered by **LangChain**, **Groq**, **RAG**, and **Streamlit**.

---

## 📁 Project Structure

```
digital_twin/
├── app.py              ← Streamlit chat UI (run this!)
├── agent.py            ← LangChain ReAct agent + memory
├── rag.py              ← PDF loading, chunking, embedding, FAISS
├── tools.py            ← Calculator, web search, Wikipedia, resume search
├── config.py           ← API keys and settings
├── requirements.txt    ← Python dependencies
├── .env.example        ← Template for your secrets
└── README.md           ← This file
```

---

## 🚀 Setup Guide (Step by Step)

### Step 1: Clone / Download the Project

```bash
cd digital_twin
```

### Step 2: Create a Virtual Environment

```bash
# Create a virtual environment (isolates project dependencies)
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

> ✅ You'll see `(venv)` in your terminal when it's active.

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

> ⏱️ This takes 2–5 minutes on first run (downloads PyTorch + models).

### Step 4: Get Your FREE Groq API Key

1. Go to **https://console.groq.com**
2. Sign up (free, no credit card)
3. Click **"Create API Key"**
4. Copy the key (starts with `gsk_...`)

### Step 5: Set Up Your Environment Variables

```bash
# Copy the template
cp .env.example .env
```

Open `.env` in any text editor and fill in:

```env
GROQ_API_KEY=gsk_your_actual_key_here
YOUR_NAME=Jane Doe
YOUR_TAGLINE=Full Stack Developer | ML Enthusiast
```

### Step 6: Run the App!

```bash
streamlit run app.py
```

Your browser will open at **http://localhost:8501** 🎉

---

## 🎮 How to Use

1. Upload your **resume PDF** in the left sidebar
2. Wait ~30 seconds for the knowledge base to build (only on first run!)
3. Start chatting! Try:
   - *"Tell me about your work experience"*
   - *"What's 23% of $450?"*
   - *"Search for the latest Python 3.13 features"*
   - *"What is FAISS?"*

---

## 🧠 Architecture Deep Dive

```
Your Resume PDF
      │
      ▼
┌─────────────────────────────┐
│         RAG Pipeline        │
│  1. PyPDFLoader → raw text  │
│  2. TextSplitter → chunks   │  ← rag.py
│  3. HuggingFace → vectors   │
│  4. FAISS → vector index    │
└─────────────────────────────┘
      │ retriever
      ▼
┌─────────────────────────────────────────────┐
│              LangChain Agent                │
│                                             │
│  User Message → [ReAct Loop]                │
│                                             │  ← agent.py
│  Thought: what do I need?                   │
│  Action:  call a tool                       │
│  Observe: tool output                       │
│  Repeat until: Final Answer                 │
└─────────────────────────────────────────────┘
      │ tools
      ▼
┌──────────┬───────────┬────────────┬──────────────┐
│ resume   │calculator │ web_search │  wikipedia   │  ← tools.py
│ _search  │           │ (DuckDuckGo│  _search     │
└──────────┴───────────┴────────────┴──────────────┘
      │
      ▼
┌─────────────────────────────┐
│      Groq LLM (free)        │
│   llama-3.3-70b-versatile   │  ← agent.py + config.py
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│   Streamlit Chat UI         │  ← app.py
└─────────────────────────────┘
```

---

## 📚 Key Concepts Explained

### What is RAG?
RAG = **Retrieval-Augmented Generation**. Instead of asking the LLM to "remember" your resume (it can't), we store your resume in a searchable database and retrieve relevant parts **at query time**, giving them to the LLM as context.

### What is a Vector Store?
Text is converted to lists of numbers (vectors) by an embedding model. Similar texts produce similar vectors. FAISS stores these vectors and can find the most similar ones to any query in milliseconds.

### What is LangChain?
LangChain is a framework that makes it easy to:
- Connect LLMs to tools (functions)
- Build agents that reason and act
- Create RAG pipelines
- Manage conversation memory

### What is the ReAct Pattern?
ReAct = **Re**asoning + **Act**ing. The LLM alternates between:
- **Thought**: planning what to do
- **Action**: calling a tool
- **Observation**: reading the result
Until it has a **Final Answer**.

---

## 🔧 Customization Tips

### Change the LLM Model
In `config.py`:
```python
GROQ_MODEL = "llama3-8b-8192"      # faster
GROQ_MODEL = "mixtral-8x7b-32768"  # longer context
```

### Add a New Tool
In `tools.py`, add:
```python
@tool
def my_new_tool(input: str) -> str:
    """Describe when and how to use this tool."""
    # your Python code here
    return result
```
Then add `my_new_tool` to the list in `get_all_tools()`.

### Customize the Agent's Personality
In `agent.py`, edit the `build_system_prompt()` function to match your tone.

---

## ❓ Troubleshooting

| Problem | Solution |
|---------|----------|
| `GROQ_API_KEY missing` | Create `.env` from `.env.example` and add your key |
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` |
| `Slow first run` | Normal! Embedding model downloads ~90MB once |
| `No results from PDF` | Make sure the PDF has selectable text (not a scanned image) |
| `Agent gives wrong info` | Add more detail to your PDF; lower `CHUNK_SIZE` in config |

---

## 🏆 Hackathon Tips

- **Impress judges**: Show the verbose ReAct trace in the terminal — it demonstrates AI reasoning visually
- **Personalize**: Add specific projects, tech stack, and achievements to your resume
- **Demo flow**: Start with "Who are you?" then "What's 20% of 500?" then a web search
- **Bonus**: Mention you implemented RAG from scratch — that's impressive!

---

*Built for the Agentic AI Hackathon | LangChain + Groq + FAISS + Streamlit*
