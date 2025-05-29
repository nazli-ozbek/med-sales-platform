import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

from app.database import get_procedure_by_name, get_doctors_by_procedure
from app.session import NegotiationSession
from app.state_detector import detect_state, detect_procedure
from textblob import TextBlob
from app.conversation_manager import ConversationManager
from app.summarizer_agent import SummarizerAgent
from app.questionnaire_manager import QuestionnaireManager

# Ortam değişkenlerini yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Model ayarları
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-2.0-flash-lite"
INDEX_FOLDER = "indexes/"

# Global objeler
chat_history = []
polarity_list = []
summarizer = SummarizerAgent()
manager = ConversationManager()
questionnaire = QuestionnaireManager()
form_completed = False
form_completed = False

last_procedure = "rhinoplasty"
faiss_index, chunks = None, None
procedure_info = None
session = None

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

        f"Sentiment Analysis:\n"
        f"- Current message polarity: {polarity:.2f}\n"
        f"- Average sentiment polarity: {avg_polarity:.2f}\n"
        f"- Conversation Summary:\n{summarizer.get_summary()}\n\n"
        "Interpretation:\n"
        "- Close to +1.0 → excited, trusting\n"
        "- Close to  0.0 → neutral, uncertain\n"
        "- Close to -1.0 → frustrated, skeptical\n"
        "Use this to adapt your tone accordingly."
        
        "\n\nAdditional Behavioral Guidelines:\n"
        "- Maintain a warm, conversational, and slightly playful tone when appropriate. You should sound natural and emotionally intelligent, like a friendly assistant rather than a formal bot.\n"
        "- If the user asks unrelated or personal questions (e.g. 'Am I beautiful?', 'Do you like me?'), respond positively and tie it gently back to the cosmetic procedure context. For example: 'Of course you are! But with this procedure, you might feel even more confident and radiant.'\n"
        "- Avoid cold or robotic answers. Use friendly expressions, occasional mild humor, and empathetic phrasing when replying.\n"
        "- Do not decline answering lighthearted or off-topic questions outright. Instead, acknowledge them warmly and subtly bring the conversation back on track.\n"
        "- Even when giving factual medical information, use human-like language. Example: Instead of 'The swelling lasts 3-5 days', say 'You'll probably notice some swelling for 3 to 5 days, but nothing to worry about — it’s totally expected and manageable.'\n"
        "- Always try to make the user feel heard, valued, and supported throughout the conversation.\n"
    )

    if session and session.doctor:
        system_prompt += f'\n\nSelected Doctor:\n- {session.doctor["name"]} ({session.doctor["specialization"]})'

        # Kullanıcının geçmiş form yanıtlarını Medical Profile olarak ekle
        if session.medical_profile:
            profile_lines = []
            for q in sorted(session.medical_profile):
                question = session.medical_profile[q]["question"]
                answer = session.medical_profile[q]["answer"]
                profile_lines.append(f"- {question} {answer}")
            system_prompt += "\n\nMedical Profile:\n" + "\n".join(profile_lines)

    if last_agent_msg:
        system_prompt += f"\nPrevious Agent Message:\n\"{last_agent_msg.strip()}\"\n"

    if current_state == "FINAL_CONFIRMATION":
        system_prompt += (
            "\n\nThe user has confirmed the agreement. "
            "Thank them warmly, say you’ll notify the doctor, and end the conversation positively. "
            "Do NOT ask any questions, just inform the patient and say goodbye. "
            "Keep it friendly, short, and professional."
        )

        if session:
            session.session_closed = True

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

# ---------------- Web API ----------------
async def handle_chat_request(chat_input):
    global last_procedure, faiss_index, chunks, procedure_info, session, form_completed

    if session and getattr(session, "session_closed", False):
        return {"response": "This conversation has been completed. Please start a new session if you have other questions."}
    if chat_input.message.strip() == "__RESET__":
        global chat_history, polarity_list, summarizer, manager, questionnaire, form_completed
        chat_history = []
        polarity_list = []
        summarizer = SummarizerAgent()
        manager = ConversationManager()
        questionnaire = QuestionnaireManager()
        form_completed = False
        procedure_info = get_procedure_by_name(last_procedure)
        session = None


        # LLM ile gerçek karşılama mesajı üret
        greeting_model = genai.GenerativeModel(CHAT_MODEL)
        greeting_prompt = (
            "You are a friendly and professional medical assistant chatbot. "
            "Begin the conversation by warmly greeting the user and clearly explaining that, before providing any medical assistance, "
            "you will ask them 8 short medical questions. "
            "Do not ask the first question yet, and do not include any system notes or acknowledgements. "
            "Your response should only be a brief introduction and explanation — no question should be asked."

        )
        greeting = greeting_model.generate_content(greeting_prompt).text.strip()

        first_q = questionnaire.get_all_questions()
        first_q_text = "\n".join(first_q)
        chat_history.append(f"Agent: {greeting}")
        chat_history.append(f"Agent: {first_q_text}")
        return {
            "response": f"{greeting}\n\nBefore we begin, please answer the following questions in a single message:\n\n{first_q_text}"}

    user_query = clean_input(chat_input.message)



    try:
        detected_procedure = detect_procedure(user_query)
        if detected_procedure == "unknown":
            detected_procedure = last_procedure

        if detected_procedure != last_procedure or faiss_index is None:
            faiss_index, chunks = load_faiss_index_and_chunks(detected_procedure)
            procedure_info = get_procedure_by_name(detected_procedure)
            session = NegotiationSession(procedure_info)
            last_procedure = detected_procedure

        last_agent_msg = chat_input.last_agent_msg or None
        if not last_agent_msg:
            for entry in reversed(chat_history):
                if entry.startswith("Agent:"):
                    last_agent_msg = entry.replace("Agent:", "").strip()
                    break

        detected_state = detect_state(user_query, last_agent_message=last_agent_msg)
        manager.update_state(detected_state)

        if detected_state == "QUESTIONNAIRE" and not questionnaire.is_complete():
            success = questionnaire.process_bulk_response(user_query)

            if success and questionnaire.is_complete():
                form_completed = True
                # cevapları dosyaya yaz
                os.makedirs("saved_forms", exist_ok=True)
                with open("saved_forms/last_questionnaire.json", "w", encoding="utf-8") as f:
                    json.dump(questionnaire.get_all_answers(), f, indent=2, ensure_ascii=False)
                if session:
                    session.set_questionnaire(questionnaire.get_all_answers())
                msg = "Thanks! Your form has been saved successfully. How can I assist you now?"
                chat_history.append(f"Agent: {msg}")
                return {"response": msg}
            else:
                missing = questionnaire.get_unanswered_questions()
                msg = "I couldn't detect all your answers. Please respond to the following questions:\n\n" + "\n".join(
                    missing)
                chat_history.append(f"Agent: {msg}")
                return {"response": msg}

        if user_query.isdigit() and manager.current_state in ("SELECT_DOCTOR", "SELECT_DOCTOR_DONE"):
            idx = int(user_query) - 1
            doctors = get_doctors_by_procedure(detected_procedure)
            if 0 <= idx < len(doctors):
                chosen_doctor = doctors[idx]
                session.set_doctor(chosen_doctor)
                if manager.current_state == "SELECT_DOCTOR":
                    manager.update_state("SELECT_DOCTOR_DONE")
                else:
                    manager.update_state("ASK_RISKS")
                answer = generate_answer(
                    model_name=CHAT_MODEL,
                    user_message=user_query,
                    context="",
                    current_state=detected_state,
                    last_agent_msg=last_agent_msg,
                    session=session
                )
                return {"response": answer}
            else:
                return {"response": "Invalid choice. Please try again."}

        if detected_state == "SELECT_DOCTOR":
            doctors = get_doctors_by_procedure(detected_procedure)
            doctor_list = "\n".join([f"{i+1}. {doc['name']} ({doc['specialization']})" for i, doc in enumerate(doctors)])
            return {"response": f"Suitable doctors for this procedure: \n \n{doctor_list}\nPlease select a doctor (enter a number)."}

        if detected_state == "ESCALATE":
            return {"response": "I'm connecting you to a specialist representative. Please hold on..."
        }

        if detected_state in ("NEGOTIATE", "ASK_PRICE", "ACCEPT_PRICE"):
            negotiation_response = session.respond(user_query, detected_state=detected_state)
            chat_history.append(f"Agent: {negotiation_response}")
            return {"response": negotiation_response}

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

        return {"response": answer}

    except Exception as e:
        print("[ERROR]", e)
        try:
            fallback_prompt = (
                "You are a polite and understanding medical AI assistant."
                " There was a temporary issue while processing the user's request. "
                "Apologize kindly and offer to connect the user with a human representative if needed."
            )
            model = genai.GenerativeModel(CHAT_MODEL)
            response = model.generate_content(
                fallback_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=150,
                )
            )
            return {"response": response.text.strip()}
        except Exception as inner_e:
            print("[ERROR]", inner_e)
            return {"response": "Üzgünüm, şu anda isteğinizi işleyemiyorum. En kısa sürede bir temsilcimiz size yardımcı olacaktır."}
