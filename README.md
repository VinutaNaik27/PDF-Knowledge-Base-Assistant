# 📄 PDF Knowledge Base Assistant

A Streamlit-based AI application that allows users to upload PDF documents, create a searchable knowledge base using FAISS vector storage, and ask questions about the uploaded content using Groq LLMs and LangChain.

## 🚀 Features

* Upload one or multiple PDF documents
* Extract text from PDFs
* Split documents into manageable chunks
* Generate embeddings using Sentence Transformers
* Store embeddings locally using FAISS
* Ask questions about uploaded documents
* Retrieve relevant document chunks automatically
* Generate context-aware answers using Groq LLM
* View source document chunks used to generate answers
* Interactive chat interface powered by Streamlit

---

## 🛠️ Technologies Used

### Frontend

* Streamlit

### Backend

* Python

### AI & LLM

* Groq API
* LangChain

### Embeddings

* Sentence Transformers
* HuggingFace Embeddings

### Vector Database

* FAISS

### PDF Processing

* PyPDF

---

## 📂 Project Structure

```text
PDF-Knowledge-Base-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .env
│
└── faiss_index/
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd PDF-Knowledge-Base-Assistant
```

### 2. Create Virtual Environment

```bash
python -m venv .venv
```

### Activate Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### Linux/Mac

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from:

https://console.groq.com/keys

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

or

```bash
py -m streamlit run app.py
```

The application will start at:

```text
http://localhost:8501
```

---

## 📖 How to Use

### Step 1: Upload PDFs

Use the sidebar to upload one or more PDF files.

### Step 2: Build Index

Click the **Build Index** button to:

* Extract text from PDFs
* Split text into chunks
* Generate embeddings
* Store vectors in FAISS

### Step 3: Ask Questions

Enter your question in the chat box.

Example:

```text
What is recycling of water?
Explain monsoon.
What are the key stakeholders in sustainability assessment?
```

### Step 4: View Sources

Expand **Source Context** to see the document chunks used to generate the answer.

---

## 🔍 Architecture

```text
PDF Files
    │
    ▼
PyPDF Text Extraction
    │
    ▼
Text Chunking
(RecursiveCharacterTextSplitter)
    │
    ▼
HuggingFace Embeddings
    │
    ▼
FAISS Vector Store
    │
    ▼
Retriever
    │
    ▼
Groq LLM
(ChatGroq)
    │
    ▼
Answer Generation
```

---

## 📦 Required Packages

```text
streamlit
python-dotenv
pypdf

langchain
langchain-core
langchain-community
langchain-text-splitters
langchain-groq
langchain-google-genai

faiss-cpu
sentence-transformers
huggingface-hub
torch
torchvision
transformers
```

---

## ⚠️ Limitations

* Works best with text-based PDFs.
* Scanned image PDFs require OCR support.
* Large PDF collections may increase indexing time.
* Requires an active Groq API key.

---

## 🔮 Future Enhancements

* OCR support for scanned PDFs
* Chat history persistence
* Multiple vector database options
* PDF summarization
* Citation generation
* User authentication
* Cloud deployment support

---

## 👨‍💻 Author

Developed using Streamlit, LangChain, FAISS, HuggingFace Embeddings, and Groq LLM.
