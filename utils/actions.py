import os
import pickle

from dotenv import load_dotenv, find_dotenv
from langchain.prompts import PromptTemplate
from langchain.retrievers import ParentDocumentRetriever
from langchain.schema import Document
from langchain.storage import InMemoryByteStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from twilio.rest import Client


load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ACCOUNT_SID   = os.getenv("ACCOUNT_SID")
AUTH_TOKEN    = os.getenv("AUTH_TOKEN")
WHATSAPP_FROM = os.getenv("WHATSAPP_FROM")  # p. ej. "whatsapp:+14155238886"

if not GROQ_API_KEY:
    raise ValueError("Define la variable de entorno GROQ_API_KEY con tu API key de Groq")

def ask_groq_chat(context: str, question_human: str) -> str:

    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.3-70b-versatile",
        max_tokens=512,
        temperature=0.2
    )

    template = """
    Sos un asistente de atención al cliente para la concesionaria de motos LiderMotos. Te caracterizás por ser conciso y directo, entendiendo que los usuarios interactúan con nosotros a través de WhatsApp y prefieren respuestas breves y claras. Pero, no te olvides siempre de saludar y ser amable. Tu objetivo es proporcionar respuestas útiles y precisas a las preguntas de los usuarios, basándote exclusivamente en la información disponible en el contexto proporcionado. En el caso que no tengas información suficiente para responder, debes ser honesto y decir que no puedes ayudar con esa consulta en puntual y ofrecerle al usuario asistencia personalizada con un asesor.

    Contexto:
    {context}

    Pregunta del cliente: {question_human}
    Respuesta del asistente:
    """

    prompt = PromptTemplate(
        input_variables=["context", "question_human"],
        template=template
    )

    prompt_input = prompt.format(context=context, question_human=question_human)

    response = llm.invoke(prompt_input)

    return response.content.strip()







if not all([ACCOUNT_SID, AUTH_TOKEN, WHATSAPP_FROM]):
    raise ValueError(
        "Debes definir las variables de entorno: "
        "ACCOUNT_SID, AUTH_TOKEN, WHATSAPP_FROM y WHATSAPP_TO"
    )

client = Client(ACCOUNT_SID, AUTH_TOKEN)

def send_whatsapp_text(body: str, to) -> None:
    """
    Envía un mensaje de WhatsApp de texto libre.
    - `body`: el contenido del mensaje que quieras enviar (string).
    Usa los números definidos en TWILIO_WHATSAPP_FROM y WHATSAPP_TO.
    """
    message = client.messages.create(
        from_=WHATSAPP_FROM,
        to=to,
        body=body
    )
    print(f"Mensaje enviado, SID: {message.sid}")




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
      


