import logging
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def process_documents(file_paths):
    """
    Process PDF and Word documents to extract text.
    
    Args:
        file_paths (list): List of file paths to process.
        
    Returns:
        list: List of extracted text documents.
    """
    documents = []
    
    for file_path in file_paths:
        try:
            if file_path.endswith(".pdf"):
                with open(file_path, "rb") as file:
                    pdf_reader = PdfReader(file)
                    text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text() or ""
                        text += page_text
                    if text.strip():
                        documents.append({"text": text, "source": file_path})
                    else:
                        logger.warning(f"No text extracted from {file_path}")
            elif file_path.endswith(".docx"):
                doc = DocxDocument(file_path)
                text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                if text.strip():
                    documents.append({"text": text, "source": file_path})
                else:
                    logger.warning(f"No text extracted from {file_path}")
            else:
                logger.warning(f"Unsupported file type: {file_path}")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    return documents