import os
import pickle
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.storage import InMemoryByteStore
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_retriever():
	BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
	VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb")
	PICKLE_PATH= os.path.join(VECTOR_DB_DIR, "byte_store.pkl")

	# Cargar el byte_store serializado
	with open(PICKLE_PATH, "rb") as f:
		byte_store: InMemoryByteStore = pickle.load(f)
    
    # Cargar el mismo embedder y carpeta donde persistió el vectordb
	embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )

	vectordb = Chroma(
        collection_name="corpus_vectordb",
        embedding_function=embedder,
        persist_directory=VECTOR_DB_DIR
    )

    # Mismo splitter que en intake
	splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

	retriever = ParentDocumentRetriever(
        vectorstore=vectordb,
        byte_store=byte_store,
        child_splitter=splitter
    )

	return retriever
