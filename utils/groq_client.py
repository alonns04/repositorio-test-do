import os
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")

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