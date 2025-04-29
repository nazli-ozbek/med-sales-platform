import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

from database import get_procedure_by_name
from negotiation.session import NegotiationSession
from state_detector import detect_state, detect_procedure
from textblob import TextBlob
from conversation_manager import ConversationManager


# Ortam değişkenlerini yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Model ayarları
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-2.0-flash"
INDEX_FOLDER = "indexes/"


# Chat geçmişi ve duygu skorları
chat_history = []
polarity_list = []

def clean_input(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def load_faiss_index_and_chunks(procedure_name: str):
    index_path = os.path.join(INDEX_FOLDER, f"{procedure_name}.index")
    chunks_path = os.path.join(INDEX_FOLDER, f"{procedure_name}_chunks.json")

    if not os.path.exists(index_path) or not os.path.exists(chunks_path):
        raise FileNotFoundError(f"{procedure_name} için gerekli dosyalar bulunamadı.")

    index = faiss.read_index(index_path)
    with open(chunks_path, "r", encoding="utf-8") as jf:
        chunks = json.load(jf)
    return index, chunks

def embed_query(query):
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_document"
    )
    vec = response["embedding"]
    return np.array(vec, dtype="float32").reshape(1, -1)

def find_relevant_chunks(index, chunks, q_vec, top_k=2):
    _, idx = index.search(q_vec, top_k)
    return [chunks[i] for i in idx[0]]

def generate_answer(model_name, user_message, context, current_state):
    model = genai.GenerativeModel(model_name)

    # Duygu analizi
    blob = TextBlob(user_message)
    polarity = blob.sentiment.polarity
    polarity_list.append(polarity)
    avg_polarity = sum(polarity_list) / len(polarity_list)

    # Konuşma geçmişini güncelle
    chat_history.append(f"User: {user_message}")
    conversation_log = "\n".join(chat_history[-6:])  # Son 3 çift (6 satır)

    system_prompt = (
        "You are a helpful and natural-sounding AI assistant helping users with medical procedures.\n"
        "You have access to background context and a classified user intent (called Detected State).\n"
        "Use these to guide your tone, reply, and conversation strategy.\n\n"

        "Detected State meanings:\n"
        "- QUIT → If user types quit finish the"
        "- LATENT_INTEREST → The user is hinting at an issue but not asking directly. Respond gently, suggest a procedure, and ask if they want more info.\n"
        "- ASK_INFO → Provide short, friendly explanation. Ask if they'd like to hear risks.\n"
        "- ASK_RISKS → List major risks clearly. Ask if they want recovery details too.\n"
        "- ASK_RECOVERY → Explain recovery timeline, pain level, activity restrictions.\n"
        "- ASK_PRICE → State the base price. Mention value or what it includes.\n"
        "- NEGOTIATE → Respond kindly. Decline or counteroffer within bounds.\n"
        "- ACCEPT → Confirm enthusiastically and offer next steps.\n"
        "- ASK_ALTERNATIVES → Suggest other treatments that could help.\n"
        "- ESCALATE → Politely explain a human will assist.\n\n"

        "Sentiment Analysis:\n"
        "- Current message polarity: {polarity:.2f}\n"
        "- Average sentiment polarity: {avg_polarity:.2f}\n"
        "Interpretation:\n"
        "- Close to +1.0 → User is excited, trusting, open.\n"
        "- Close to  0.0 → User is neutral or uncertain.\n"
        "- Close to -1.0 → User is skeptical, emotional, or frustrated.\n"
        "Use this to adapt your tone.\n\n"

        "Always include a soft follow-up question (unless user said 'quit') to keep the conversation flowing."
    )


    prompt = (
        f"{system_prompt}\n\n"
        f"Detected State: {current_state}\n"
        f"Context:\n{context}\n\n"
        f"Conversation History:\n{conversation_log}\n"
        f"User Message: {user_message}\n"
        f"Agent Response:"
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=256,
        )
    )

    response_text = response.text.strip()
    chat_history.append(f"Agent: {response_text}")
    return response_text

# ---------------- Ana Program ----------------
def main():
    print("=== Çoklu Prosedür Destekli RAG Chatbot ===")
    print("Soru sorabilirsiniz. Çıkmak için 'quit' yazın.\n")

    last_procedure = "rhinoplasty"
    faiss_index, chunks = load_faiss_index_and_chunks(last_procedure)
    procedure_info = get_procedure_by_name(last_procedure)
    manager = ConversationManager()

    while True:
        raw_query = input("Soru: ").strip()

        if not raw_query:
            continue

        user_query = clean_input(raw_query)

        if user_query.lower() in ["quit", "exit"]:
            print("Görüşmek üzere!")
            break

        # 1. Prosedür tespiti
        detected_procedure = detect_procedure(user_query)
        if detected_procedure == "unknown":
            detected_procedure = last_procedure
        print(f"[DEBUG] Detected Procedure: {detected_procedure}")

        if detected_procedure != last_procedure:
            faiss_index, chunks = load_faiss_index_and_chunks(detected_procedure)
            procedure_info = get_procedure_by_name(detected_procedure)
            last_procedure = detected_procedure

        # 2. State tespiti
        detected_state = detect_state(user_query)
        print(f"[DEBUG] Detected State: {detected_state}")
        manager.update_state(detected_state)

        # 3. ESCALATE durumunda özel yönlendirme
        if detected_state == "ESCALATE":
            print("Bu konuda seni uzman bir temsilciye yönlendiriyorum. Lütfen bekle...")
            continue

        # 4. Sorgu gömme + context bulma
        query_vec = embed_query(user_query)
        best_chunks = find_relevant_chunks(faiss_index, chunks, query_vec, top_k=2)
        context_text = "\n\n".join(best_chunks)

        # 5. LLM üzerinden cevap üretimi
        answer = generate_answer(
            model_name=CHAT_MODEL,
            user_message=user_query,
            context=context_text,
            current_state=detected_state
        )

        print("\nCevap:\n", answer, "\n")

if __name__ == "__main__":
    main()