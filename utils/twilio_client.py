import os
from twilio.rest import Client
from dotenv import load_dotenv
import os

load_dotenv()  # Carga las variables del archivo .env

ACCOUNT_SID   = os.getenv("ACCOUNT_SID")
AUTH_TOKEN    = os.getenv("AUTH_TOKEN")
WHATSAPP_FROM = os.getenv("WHATSAPP_FROM")  # p. ej. "whatsapp:+14155238886"
WHATSAPP_TO   = os.getenv("WHATSAPP_TO")    # p. ej. "whatsapp:+54911XXXXYYYY"

if not all([ACCOUNT_SID, AUTH_TOKEN, WHATSAPP_FROM, WHATSAPP_TO]):
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
