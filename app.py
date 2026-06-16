import os
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
load_dotenv()

# Load environment variables
GROQ_API_KEY = (
    os.getenv("GROQ_API_KEY")
    or st.secrets.get("GROQ_API_KEY")
)
# Models
model="llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# FAISS folder
DB_PATH = "faiss_index"

# Streamlit config
st.set_page_config(
    page_title="PDF Assistant",
    page_icon="📄",
    layout="wide"
)


# ---------------- PDF PROCESSING ---------------- #

def process_pdfs(pdf_docs):
    text = ""

    for pdf in pdf_docs:
        try:
            reader = PdfReader(pdf)

            for page in reader.pages:
                content = page.extract_text()

                if content:
                    text += content

        except Exception as e:
            st.sidebar.error(f"Error reading {pdf.name}: {e}")
            return False

    if not text.strip():
        st.sidebar.error(
            "No text found in PDFs. They may be scanned images."
        )
        return False

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

        vector_store = FAISS.from_texts(
            texts=chunks,
            embedding=embeddings
        )

        vector_store.save_local(DB_PATH)

        if "qa_chain" in st.session_state:
            del st.session_state["qa_chain"]

        return True

    except Exception as e:
        st.sidebar.error(f"Error building vector store: {e}")
        return False


# ---------------- QA CHAIN ---------------- #

def get_qa_chain():

    if "qa_chain" not in st.session_state:

        # API KEY CHECK
        if not GROQ_API_KEY:
            st.error("Groq API Key not found.")
            return None

        try:
            embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL
            )

            db = FAISS.load_local(
                DB_PATH,
                embeddings,
                allow_dangerous_deserialization=True
            )

            template = """
You are a professional assistant.

Answer ONLY from the provided context.

If answer is unavailable, say:
"Answer is not available in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

            prompt = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )

            # Correct Groq initialization
            llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name=model
)

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(
                    search_kwargs={"k": 4}
                ),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )

            st.session_state.qa_chain = qa_chain

        except Exception as e:
            st.error(f"Error creating QA chain: {e}")
            return None

    return st.session_state.qa_chain


# ---------------- MAIN APP ---------------- #

def main():

    st.title("📄 PDF Knowledge Base Assistant")

    st.write(
        "Upload PDFs, build the index, and ask questions."
    )

    st.markdown("---")

    # Session State
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "source_docs" not in st.session_state:
        st.session_state.source_docs = None

    # Sidebar
    with st.sidebar:

        st.header("Document Management")

        uploaded_files = st.file_uploader(
            "Upload PDF Files",
            type="pdf",
            accept_multiple_files=True
        )

        if st.button("Build Index", use_container_width=True):

            if uploaded_files:

                with st.spinner("Processing PDFs..."):

                    success = process_pdfs(uploaded_files)

                if success:
                    st.success("Knowledge base ready!")

            else:
                st.error("Upload at least one PDF.")

    # Display Chat History
    for msg in st.session_state.messages:

        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # User Input
    if prompt := st.chat_input(
        "Ask something about your documents..."
    ):

        if not os.path.exists(DB_PATH):
            st.error(
                "Please upload and index PDFs first."
            )
            return

        with st.chat_message("user"):
            st.write(prompt)

        st.session_state.messages.append(
            {
                "role": "user",
                "content": prompt
            }
        )

        with st.spinner("Searching documents..."):

            chain = get_qa_chain()

            if chain:

                try:

                    response = chain.invoke(
                        {"query": prompt}
                    )

                    answer = response["result"]

                    st.session_state.source_docs = (
                        response["source_documents"]
                    )

                    with st.chat_message("assistant"):
                        st.write(answer)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer
                        }
                    )

                    st.rerun()

                except Exception as e:
                    st.error(f"Query Error: {e}")

    # Source Chunks
    if st.session_state.source_docs:

        st.markdown("---")

        with st.expander("🔍 Source Context"):

            for i, doc in enumerate(
                st.session_state.source_docs
            ):

                st.markdown(f"### Chunk {i+1}")

                st.info(doc.page_content)


if __name__ == "__main__":
    main()

