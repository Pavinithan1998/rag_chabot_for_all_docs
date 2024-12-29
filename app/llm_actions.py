import os
import base64
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

chat_model = ChatOpenAI(
    api_key = OPENAI_API_KEY,
    model = "gpt-4o"
)

def get_image_description(image_bytes: bytes):
    image_data = base64.b64encode(image_bytes).decode("utf-8")
    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe the image"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}},
        ],
    )
    
    response = chat_model.invoke([message])
    return response.content