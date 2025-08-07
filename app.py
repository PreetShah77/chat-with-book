import streamlit as st
import os
import logging.config
import yaml
import tempfile
from dotenv import load_dotenv
from document_processor import process_documents
from rag_pipeline import create_vector_store, ask_question
from openai import OpenAI
from langchain_core.exceptions import LangChainException

with open("logging_config.yaml", "r") as f:
    config = yaml.safe_load(f)
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://chatapi.akash.network/api/v1")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "processing_status" not in st.session_state:
    st.session_state.processing_status = None

st.set_page_config(page_title="RAG Document Chatbot", page_icon="📚")
st.title("📚 Bookbot")
st.write("Upload PDFs or Word documents and ask questions about their content.")

with st.sidebar:
    st.header("Configuration")
    api_key = OPENAI_API_KEY
    base_url = OPENAI_BASE_URL
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
    if base_url:
        os.environ["OPENAI_BASE_URL"] = base_url
    else:
        st.warning("Please enter a valid OpenAI Base URL.", icon="⚠️")
    
    chunk_size = 512
    k = 3
    uploaded_files = st.file_uploader("Upload Documents", type=["pdf", "docx"], accept_multiple_files=True)
    process_button = st.button("Process Documents")
    clear_button = st.button("Clear History")

if not api_key or not api_key.startswith("sk-"):
    st.error("Invalid or missing OpenAI API key. Please provide a valid key.")
    logger.error("Invalid or missing API key")
    st.stop()

# Initialize OpenAI client
try:
    client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),
        base_url=os.environ.get("OPENAI_BASE_URL")
    )
except Exception as e:
    st.error(f"Failed to initialize OpenAI client: {str(e)}")
    logger.error(f"OpenAI client initialization error: {str(e)}")
    st.stop()

# Process uploaded documents
if process_button and uploaded_files:
    try:
        with st.spinner("Processing documents..."):
            max_file_size = 10 * 1024 * 1024  
            for file in uploaded_files:
                if file.size > max_file_size:
                    st.error(f"File {file.name} exceeds 10MB limit.")
                    logger.warning(f"File {file.name} exceeds size limit: {file.size} bytes")
                    st.stop()
            
            with tempfile.TemporaryDirectory() as temp_dir:
                file_paths = []
                for file in uploaded_files:
                    file_path = os.path.join(temp_dir, file.name)
                    with open(file_path, "wb") as f:
                        f.write(file.read())
                    file_paths.append(file_path)
                
                # Process documents
                documents = process_documents(file_paths)
                if not documents:
                    st.error("No valid text extracted from documents.")
                    logger.warning("No text extracted from uploaded documents")
                    st.stop()
                
                # Create vector store
                vector_store = create_vector_store(documents, client, chunk_size)
                if vector_store is None:
                    st.error("Failed to create vector store. Please check logs for details.")
                    logger.error("Vector store creation failed")
                    st.stop()
                st.session_state.vector_store = vector_store
                st.session_state.processing_status = "Documents processed successfully!"
                st.success(st.session_state.processing_status)
                logger.info("Documents processed and vector store created")
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        logger.error(f"Document processing error: {str(e)}")
        st.session_state.processing_status = None
        st.stop()
if clear_button:
    st.session_state.chat_history = []
    st.session_state.vector_store = None
    st.session_state.processing_status = None
    st.success("Chat history and vector store cleared.")
    logger.info("Chat history and vector store cleared")

st.subheader("Chat with Your Documents")
user_input = st.text_input("Ask a question:", "")
if st.button("Submit Question") and user_input:
    if st.session_state.vector_store is None:
        st.error("Please process documents before asking questions.")
        logger.warning("Question submitted without processed documents")
    else:
        try:
            with st.spinner("Generating answer..."):
                answer = ask_question(st.session_state.vector_store, user_input, client, k)
                st.session_state.chat_history.append({"role": "user", "content": user_input})
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
                logger.info(f"Question: {user_input} | Answer: {answer}")
        except LangChainException as e:
            st.error(f"Error generating answer: {str(e)}")
            logger.error(f"RAG pipeline error: {str(e)}")
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            logger.error(f"Unexpected error: {str(e)}")

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if st.session_state.processing_status:
    st.info(st.session_state.processing_status)