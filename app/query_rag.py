import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

from database import get_procedure_by_name, get_doctors_by_procedure
from session import NegotiationSession
from state_detector import detect_state, detect_procedure
from textblob import TextBlob
from conversation_manager import ConversationManager
from summarizer_agent import SummarizerAgent
from questionnaire_manager import QuestionnaireManager


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
summarizer = SummarizerAgent()
questionnaire = QuestionnaireManager()
form_completed = False

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


def generate_answer(model_name, user_message, context, current_state, last_agent_msg=None, session=None):
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
        "You are a helpful, natural, and emotionally aware AI assistant helping users with medical procedures.\n"
        "The conversation may involve information requests, price discussions, emotional concerns, or appointment confirmations.\n\n"

        "Before assisting the user, a short medical intake form must be completed (e.g., full name, age, allergies, etc.).\n"
        "The form is managed by the system. DO NOT ask these questions yourself.\n"
        "Until the form is fully completed, if the user asks unrelated things, gently remind them to continue answering the form.\n\n"

        "Once the form is completed, you will be able to assist based on:\n"
        "- The user's detected intent (called Detected State)\n"
        "- Background information (Context)\n"
        "- Conversation summary\n"
        "- User’s emotional tone (sentiment polarity)\n\n"

        "Detected State meanings:\n"
        "- QUESTIONNAIRE → The form is still being answered.\n"
        "- LATENT_INTEREST → User hints dissatisfaction (e.g. 'I hate my nose'). Gently suggest a relevant procedure. Then ask if want to get any info\n"
        "- ASK_INFO → Give short explanation. Then ask the user if they’d like to see available doctors for this procedure.\n"
        "- SELECT_DOCTOR → If the assistant has listed doctors and the user replies with a number (e.g., '1', '2'), interpret it as selecting a doctor. Then confirm selection and continue to the next step.\n"
        "- SELECT_DOCTOR_DONE → A doctor has just been selected. Confirm this and politely ask the user if they would like to hear about the procedure’s potential risks.\n"
        "- ASK_RISKS → List main risks. Then ask about recovery.\n"
        "- ASK_RECOVERY → Explain healing time and restrictions.\n"
        "- ASK_PRICE → Say the base price and value.\n"
        "- NEGOTIATE → Counter politely within the allowed range.\n"
        "- ACCEPT_PRICE → Confirm agreement and next steps.\n"
        "- ASK_ALTERNATIVES → Offer complementary options.\n"
        "- ESCALATE → Forward to human rep politely.\n"
        "- CONSULT_BOOKED → Confirm appointment and close politely.\n\n"

        f"Sentiment Analysis:\n"
        f"- Current message polarity: {polarity:.2f}\n"
        f"- Average sentiment polarity: {avg_polarity:.2f}\n"
        f"- Conversation Summary:\n{summarizer.get_summary()}\n\n"
        "Interpretation:\n"
        "- Close to +1.0 → excited, trusting\n"
        "- Close to  0.0 → neutral, uncertain\n"
        "- Close to -1.0 → frustrated, skeptical\n"
        "Use this to adapt your tone accordingly."
    )
    if session.doctor:
        system_prompt += f'\n\nSelected Doctor:\n- Name: {session.doctor["name"]} ({session.doctor["specialization"]})'

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
    last_procedure = "rhinoplasty"
    faiss_index, chunks = load_faiss_index_and_chunks(last_procedure)
    procedure_info = get_procedure_by_name(last_procedure)
    session = NegotiationSession(procedure_info)
    manager = ConversationManager()

    greeting_model = genai.GenerativeModel(CHAT_MODEL)
    greeting_prompt = (
        "You are a friendly and professional medical assistant chatbot. "
        "Start the conversation by politely greeting the user and explaining that you will ask 8 short medical questions "
        "before providing assistance. Speak directly to the user, and do not include any system notes or acknowledgements."
    )
    greeting = greeting_model.generate_content(greeting_prompt).text.strip()
    print("\nAgent:", greeting)

    # Greet cevabını history'e ekleme — çünkü bu bir soru değil
    first_q = questionnaire.get_next_question()
    print("Agent:", first_q)
    chat_history.append(f"Agent: {first_q}")

    global form_completed

    while True:
        raw_query = input("Soru: ").strip()
        if not raw_query:
            continue

        user_query = clean_input(raw_query)
        if user_query.lower() in ["quit", "exit"]:
            print("Agent: Görüşmek üzere!")
            break

        try:
            detected_procedure = detect_procedure(user_query)
            if detected_procedure == "unknown":
                detected_procedure = last_procedure
            print(f"[DEBUG] Detected Procedure: {detected_procedure}")

            if detected_procedure != last_procedure:
                faiss_index, chunks = load_faiss_index_and_chunks(detected_procedure)
                procedure_info = get_procedure_by_name(detected_procedure)
                session = NegotiationSession(procedure_info)
                last_procedure = detected_procedure
            print(f"[DEBUG] Detected Procedure: {detected_procedure}")

            last_agent_msg = next(
                (entry.replace("Agent:", "").strip() for entry in reversed(chat_history) if entry.startswith("Agent:")),
                None
            )

            detected_state = detect_state(user_query, last_agent_message=last_agent_msg)
            print(f"[DEBUG] Detected State: {detected_state}")
            manager.update_state(detected_state)

            if detected_state == "QUESTIONNAIRE" and not questionnaire.is_complete():
                questionnaire.answer_current_question(user_query)
                if questionnaire.is_complete():
                    form_completed = True
                    msg = "Thanks! How can I help you now?"
                    print("✅ Medical intake form completed.")
                    print("Agent:", msg)
                    chat_history.append(f"Agent: {msg}")
                else:
                    next_q = questionnaire.get_next_question()
                    print("Agent:", next_q)
                    chat_history.append(f"Agent: {next_q}")
                continue

            # 1. Kullanıcı doktor seçtiyse, önce bunu işle:
            if user_query.isdigit() and manager.current_state in ("SELECT_DOCTOR", "SELECT_DOCTOR_DONE"):
                idx = int(user_query) - 1
                doctors = get_doctors_by_procedure(detected_procedure)
                if 0 <= idx < len(doctors):
                    chosen_doctor = doctors[idx]
                    print(f"Agent: Dr. {chosen_doctor['name']} seçildi. Devam edebiliriz.")
                    session.set_doctor(chosen_doctor)
                    # ❗ Yeni state: SELECT_DOCTOR_DONE
                    # 🌟 STATE geçişini güncelle
                    if manager.current_state == "SELECT_DOCTOR":
                        manager.update_state("SELECT_DOCTOR_DONE")
                    else:
                        manager.update_state("ASK_RISKS")

                    # ❗ LLM'e danış: Riskleri duymak ister misiniz?
                    answer = generate_answer(
                        model_name=CHAT_MODEL,
                        user_message=user_query,
                        context="",
                        current_state=detected_state,
                        last_agent_msg=last_agent_msg,
                        session=session
                    )
                    print("\nAgent:\n", answer, "\n")
                else:
                    print("Agent: Geçersiz seçim. Lütfen tekrar deneyin.")
                continue

            # 2. Eğer doktor seçilmediyse ve SELECT_DOCTOR state'ine yeni girdiysek, listeyi göster:
            if detected_state == "SELECT_DOCTOR":
                doctors = get_doctors_by_procedure(detected_procedure)
                print("Agent: Bu prosedür için uygun doktorlar:")
                for i, doc in enumerate(doctors, 1):
                    print(f"{i}. Dr. {doc['name']} ({doc['specialization']})")
                print("Lütfen bir doktor seçin (numara girin).")
                continue

            if detected_state == "ESCALATE":
                print("Agent: I'm forwarding you to a human representative.")
                continue

            if detected_state in ("NEGOTIATE", "ASK_PRICE", "ACCEPT_PRICE"):
                negotiation_response = session.respond(user_query)
                print("Agent:", negotiation_response)
                continue

            if detected_state == "CONSULT_BOOKED":
                print("Agent: Your consultation is confirmed. Take care!")
                break

            query_vec = embed_query(user_query)
            best_chunks = find_relevant_chunks(faiss_index, chunks, query_vec, top_k=2)
            context_text = "\n\n".join(best_chunks)

            answer = generate_answer(
                model_name=CHAT_MODEL,
                user_message=user_query,
                context=context_text,
                current_state=detected_state,
                last_agent_msg=last_agent_msg,
                session=session
            )

            print("\nAgent:\n", answer, "\n")

        except Exception as e:
            print("[HATA]:", e)
            print("Agent: Sorry, there was a problem. A human representative will assist you shortly.")

if __name__ == "__main__":
    main()