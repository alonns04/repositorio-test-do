from utils.retriever import get_retriever
from utils.groq_client import ask_groq_chat
from flask import Flask, request, Response
from utils.twilio_client import send_whatsapp_text
import logging

app = Flask(__name__)

@app.route("/whatsapp", methods=["POST"])
def whatsapp_webhook():
    from_number = request.values.get("From")
    print(f"Número del remitente: {from_number}")
    question_human = request.values.get("Body")

    if not question_human:
        logging.error("No se recibió 'Body' en el POST de Twilio")
        return Response("Missing message body", status=400)

    try:
        retriever = get_retriever()
        recovered_documents = retriever.invoke(question_human)
        context = "\n\n".join(doc.page_content for doc in recovered_documents)
        response = ask_groq_chat(context=context, question_human=question_human)
        send_whatsapp_text(body=f"{response}", to=from_number)
    except Exception as e:
        logging.exception("Error en procesamiento del mensaje de WhatsApp")
        return Response("Error interno del servidor", status=500)

    return Response(status=200)

if __name__ == "__main__":
    # Ejecuta el servidor en el puerto 5000
    app.run(host="0.0.0.0", port=8000)