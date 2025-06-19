import os
import pickle
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain.retrievers import ParentDocumentRetriever

def run_intake():
	BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../src"))
	CORPUS_PATH = os.path.join(BASE_DIR, "corpus.txt")
	VECTOR_DB_DIR = os.path.join(BASE_DIR, "vectordb")

	with open(CORPUS_PATH, "r", encoding="utf-8") as f:
		corpus = f.read()

	# Dividir en secciones usando el separador '---'
	sections = [sec.strip() for sec in corpus.split("\n---\n") if sec.strip()]
	
	# Preparar lista de documentos con parent_id
	documents = [Document(page_content=chunk) for chunk in sections]
	
	# TextSplitter con ParentDocument: mantiene parent_id en cada chunk
	splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50,
    )

	os.makedirs(VECTOR_DB_DIR, exist_ok=True)

	# Inicializar embeddings con HuggingFaceBgeEmbeddings
	embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={"device": "cpu"}
    )

	# Crear o cargar Chroma usando el embedder configurado
	vectordb = Chroma(
		collection_name="corpus_vectordb",
		embedding_function=embedder,
		persist_directory=VECTOR_DB_DIR
	)

	byte_store = InMemoryByteStore()

	retriever = ParentDocumentRetriever(
		vectorstore=vectordb,
		byte_store=byte_store,
		child_splitter=splitter
	)

	retriever.add_documents(documents)

	# Persistimos el vector store que contiene la relación padre-hijo para ser reutilizado
	byte_store = retriever.byte_store
	ruta_pickle = os.path.join(VECTOR_DB_DIR, "byte_store.pkl")
	with open(ruta_pickle, "wb") as f:
		pickle.dump(byte_store, f)
	print(f"ByteStore serializado en {ruta_pickle}")

	# Solo imprimimos cuántos padres y cuántos hijos hay en memoria
	all_data = vectordb._collection.get(include=["metadatas"])
	num_chunks = len(all_data["metadatas"])
	print(f"\n\nVector store creado en ../src/vectordb/corpus_vectordb")
	print(f"Poblado con {len(documents)} documentos padres y {num_chunks} chunks hijos \n\n")

if __name__ == "__main__":
	run_intake()