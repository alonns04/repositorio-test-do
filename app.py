from utils.actions import get_retriever
from utils.actions import ask_groq_chat
from flask import Flask, request, Response
from utils.actions import send_whatsapp_text
import logging

logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    from_number = request.values.get("From")
    logging.debug(f"Número del remitente: {from_number}")

    if not from_number:
        logging.error("No se recibió 'From' en el POST de Twilio")
        return Response("Missing 'From' parameter", status=400)

    question_human = request.values.get("Body")
    logging.debug(f"Mensaje recibido: {question_human}")

    if not question_human:
        logging.error("No se recibió 'Body' en el POST de Twilio")
        return Response("Missing 'Body' parameter", status=400)

    try:
        logging.debug("Obteniendo retriever...")
        retriever = get_retriever()
        logging.debug("Ejecutando retriever.invoke()...")
        recovered_documents = retriever.invoke(question_human)
        logging.debug(f"Documentos recuperados: {len(recovered_documents)}")

        context = "\n\n".join(doc.page_content for doc in recovered_documents)
        logging.debug(f"Contexto construido: {context[:200]}...")  # Muestra primeros 200 chars

        logging.debug("Preguntando a ask_groq_chat...")
        response = ask_groq_chat(context=context, question_human=question_human)
        logging.debug(f"Respuesta generada: {response}")
        print(response)
        logging.debug("Enviando texto por WhatsApp...")
        send_whatsapp_text(body=f"{response}", to=from_number)
        logging.debug("Texto enviado exitosamente.")
    except Exception as e:
        logging.exception("Error en procesamiento del mensaje de WhatsApp")
        return Response("Error interno del servidor", status=500)

    return Response("Mensaje procesado correctamente", status=200)

if __name__ == "__main__":
    # Ejecuta el servidor en el puerto 8080 (como configuraste)
    app.run(host="0.0.0.0", port=8080)
