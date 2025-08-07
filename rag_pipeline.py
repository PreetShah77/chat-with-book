import logging
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)

class AkashChatEmbeddings(Embeddings):
    """
    Custom embedding class for AkashChat API.
    """
    def __init__(self, client, model="BAAI-bge-large-en-v1-5"):
        self.client = client
        self.model = model

    def embed_documents(self, texts):
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts (list): List of text strings to embed.
            
        Returns:
            list: List of embedding vectors.
        """
        try:
            response = self.client.embeddings.create(input=texts, model=self.model)
            return [item.embedding for item in response.data]
        except Exception as e:
            logger.error(f"Error generating embeddings for documents: {str(e)}")
            raise

    def embed_query(self, text):
        """
        Generate embedding for a single query text.
        
        Args:
            text (str): Query text to embed.
            
        Returns:
            list: Embedding vector for the query.
        """
        try:
            response = self.client.embeddings.create(input=[text], model=self.model)
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding for query: {str(e)}")
            raise

class AkashChatLLM(Runnable):
    """
    Custom LLM class for AkashChat API, implementing the Runnable interface.
    """
    def __init__(self, client, model="Meta-Llama-3-3-70B-Instruct"):
        self.client = client
        self.model = model

    def invoke(self, input, config=None, **kwargs):
        """
        Invoke the LLM with the given input (context and query).
        
        Args:
            input (str): Formatted prompt with context and question.
            config (dict, optional): Configuration parameters (not used).
            
        Returns:
            str: Generated answer.
        """
        try:
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": str(input)}  # Ensure input is string
            ]
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error in LLM invocation: {str(e)}")
            raise

    def __call__(self, *args, **kwargs):
        """
        Make the class callable to satisfy Runnable interface.
        """
        return self.invoke(*args, **kwargs)

def create_vector_store(documents, client, chunk_size):
    """
    Create a vector store from documents using embeddings.
    
    Args:
        documents (list): List of document dictionaries with 'text' and 'source'.
        client: OpenAI client instance.
        chunk_size (int): Size of text chunks.
        
    Returns:
        Chroma: Vector store with embedded documents.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200,
            length_function=len,
            add_start_index=True
        )
        
        langchain_docs = [
            Document(page_content=doc["text"], metadata={"source": doc["source"]})
            for doc in documents
        ]
        
        splits = text_splitter.split_documents(langchain_docs)
        if not splits:
            logger.error("No valid document chunks created")
            return None
        
        embeddings = AkashChatEmbeddings(client=client, model="BAAI-bge-large-en-v1-5")
        
        vector_store = Chroma.from_documents(
            documents=splits,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        return None

def ask_question(vector_store, query, client, k, model="Meta-Llama-3-3-70B-Instruct"):
    """
    Ask a question using the RAG pipeline.
    
    Args:
        vector_store: Chroma vector store.
        query (str): User query.
        client: OpenAI client instance.
        k (int): Number of documents to retrieve.
        model (str): Chat model name (default: Meta-Llama-3-3-70B-Instruct).
        
    Returns:
        str: Generated answer.
    """
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        
        prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

        {context}

        Question: {question}
        Answer: """
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        llm = AkashChatLLM(client=client, model=model)
        
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        
        result = qa_chain.invoke({"query": query})
        answer = result["result"]
        
        sources = [doc.metadata["source"] for doc in result["source_documents"]]
        logger.info(f"Retrieved documents: {sources}")
        
        return answer
    except Exception as e:
        logger.error(f"Error in RAG pipeline: {str(e)}")
        raise