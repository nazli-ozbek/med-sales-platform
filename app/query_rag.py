import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

from database import get_procedure_by_name
from session import NegotiationSession
from state_detector import detect_state, detect_procedure
from textblob import TextBlob
from conversation_manager import ConversationManager
from summarizer_agent import SummarizerAgent


# Ortam değişkenlerini yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Model ayarları
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-2.0-flash-lite"
INDEX_FOLDER = "indexes/"


# Chat geçmişi ve duygu skorları
chat_history = []
polarity_list = []
summarizer = SummarizerAgent()

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

def generate_answer(model_name, user_message, context, current_state, last_agent_msg=None):
    model = genai.GenerativeModel(model_name)

    # Duygu analizi
    blob = TextBlob(user_message)
    polarity = blob.sentiment.polarity
    polarity_list.append(polarity)
    avg_polarity = sum(polarity_list) / len(polarity_list)

    chat_history.append(f"User: {user_message}")
    summarizer.update_summary(chat_history)

    # Prompt içeriği
    system_prompt = (
        "You are a helpful and natural-sounding AI assistant helping users with medical procedures.\n"
        "However, before you can provide any information, you are required to complete a short medical intake questionnaire with the user.\n"
        "You MUST ask the following questions one-by-one and record each answer. Do not skip or rush. Do not provide any information or assistance until this form is fully completed.\n"
        "Ask questions sequentially and all with a similar format. Example: 1. What is your full name?\n"
        "Once the questionnaire is complete, ask what does the customer need help with.\n"
        "Here are the required questions:\n"
        "1. Full Name\n"
        "2. Age\n"
        "3. Allergies (if any)\n"
        "4. Expected date for the surgery\n"
        "5. Any contagious diseases (e.g. Hepatitis B, Hepatitis C, HIV)\n"
        "6. Ongoing health problems or medications (e.g. HRT, thyroid, diabetes)\n"
        "7. Height and weight\n"
        "8. Previous surgeries\n\n"
    
        "Ask each question clearly and wait for the user's answer before moving on. After all 8 questions have been answered, thank the user and smoothly transition to regular medical assistance.\n"
        "Until the questionnaire is complete, DO NOT answer any other questions from the user.\n\n"
    
        "Once the questionnaire is completed, you will also have access to:\n"
        "- The user's current intent (called Detected State),\n"
        "- Background information (Context),\n"
        "- Conversation summary so far,\n"
        "- The user's emotional tone (via sentiment polarity).\n\n"

        "Detected State meanings:\n"
        "- QUESTIONNAIRE → The user is in the intake phase and is answering personal background questions (name, age, allergies, medical history, etc.).\n" 
        "- LATENT_INTEREST → User is hinting but not asking directly. Gently suggest a procedure and ask if they'd like more info.\n"
        "- ASK_INFO → Give short helpful explanation. Ask if they'd like to hear about risks.\n"
        "- ASK_RISKS → Clearly list main risks. Ask if they want to know about recovery too.\n"
        "- ASK_RECOVERY → Talk about healing time, restrictions, etc.\n"
        "- ASK_PRICE → State the price. Mention added value if appropriate.\n"
        "- NEGOTIATE → Politely decline or counter within allowed range.\n"
        "- ACCEPT_PRICE → If the price is settled, confirm and offer next steps.\n"
        "- ASK_ALTERNATIVES → Suggest complementary procedures.\n"
        "- ESCALATE → Say a human rep will assist soon.\n"
        "- CONSULT_BOOKED → The user has booked a consultation or confirmed an appointment. Politely confirm and end the conversation without further questions.\n"
    

        f"Sentiment Analysis:\n"
        f"- Current message polarity: {polarity:.2f}\n"
        f"- Average sentiment polarity: {avg_polarity:.2f}\n"
        f"- Conversation Summary:\n{summarizer.get_summary()}\n\n"
        "Interpretation:\n"
        "- Close to +1.0 → excited, trusting\n"
        "- Close to  0.0 → neutral, uncertain\n"
        "- Close to -1.0 → frustrated, skeptical\n"
        "Use this to adapt your tone accordingly.\n"
    )
    if last_agent_msg:
        system_prompt += f"\nPrevious Agent Message:\n\"{last_agent_msg.strip()}\"\n"

    # quit/exit özel durumu için son talimat
    if user_message.lower() in ["quit", "exit"]:
        system_prompt += "\n\nThe user wants to end the conversation. Respond politely and do not ask any further questions."
    else:
        system_prompt += "\n\nAlways end your reply with a soft, relevant follow-up question to keep the conversation going."

    # Final prompt
    prompt = (
        f"{system_prompt}\n\n"
        f"Detected State: {current_state}\n"
        f"Context:\n{context}\n\n"
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
    session = NegotiationSession(procedure_info)

    greeting_prompt = (
        "You are a friendly and professional medical assistant. Greet the user warmly and friendly and explain that "
        "you will first ask a few short questions to better understand their medical background before providing help."
        ", be really friendly.\n"
    )
    model = genai.GenerativeModel(CHAT_MODEL)
    greeting_response = model.generate_content(
        greeting_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.3,
            max_output_tokens=150,
        )
    )
    greeting_text = greeting_response.text.strip()
    print("\nAgent:", greeting_text, "\n")
    chat_history.append(f"Agent: {greeting_text}")
    while True:
        raw_query = input("Soru: ").strip()

        if not raw_query:
            continue

        user_query = clean_input(raw_query)

        if user_query.lower() in ["quit", "exit"]:
            print("Goodbye!")
            break
        try:
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
            last_agent_msg = None
            for entry in reversed(chat_history):
                if entry.startswith("Agent:"):
                    last_agent_msg = entry.replace("Agent:", "").strip()
                    break

            detected_state = detect_state(user_query, last_agent_message=last_agent_msg)
            print(f"[DEBUG] Detected State: {detected_state}")
            manager.update_state(detected_state)


            # 3. ESCALATE durumunda özel yönlendirme
            if detected_state == "ESCALATE":
                print("Bu konuda seni uzman bir temsilciye yönlendiriyorum. Lütfen bekle...")
                continue

            if detected_state in("NEGOTIATE" , "ASK_PRICE", "ACCEPT_PRICE") :
                negotiation_response = session.respond(user_query)
                print("Cevap:", negotiation_response)
                continue

            if detected_state == "CONSULT_BOOKED":
                print("Görüşmek üzere!")
                break


            # 4. Sorgu gömme + context bulma
            query_vec = embed_query(user_query)
            best_chunks = find_relevant_chunks(faiss_index, chunks, query_vec, top_k=2)
            context_text = "\n\n".join(best_chunks)

            # 5. LLM üzerinden cevap üretimi
            answer = generate_answer(
                model_name=CHAT_MODEL,
                user_message=user_query,
                context=context_text,
                current_state=detected_state,
                last_agent_msg = last_agent_msg
            )

            print("\nCevap:\n", answer, "\n")



        except Exception as e:
            print(e)
            fallback_prompt = (
                "You are a polite and understanding medical AI assistant." 
                "There was a temporary issue while processing the user's request. "
                "Apologize kindly and offer to connect the user with a human representative if needed."
            )
            try:
                model = genai.GenerativeModel(CHAT_MODEL)
                response = model.generate_content(
                    fallback_prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.3,
                        max_output_tokens=150,
                    )
                )
                print("\nCevap:\n", response.text.strip(), "\n")
            except Exception as inner_e:
                print(inner_e)
                print(
                    "\nCevap:\nÜzgünüm, şu anda isteğinizi işleyemiyorum. En kısa sürede bir temsilcimiz size yardımcı olacaktır.\n")


if __name__ == "__main__":
    main()