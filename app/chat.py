from fastapi import APIRouter
from pydantic import BaseModel
from app.chat_service import handle_chat_request

router = APIRouter()

# Gelen veriyi tanımlayan model
class ChatInput(BaseModel):
    message: str
    last_agent_msg: str = ""

# POST /chat isteği
@router.post("/")
async def chat(chat_input: ChatInput):
    return await handle_chat_request(chat_input)
