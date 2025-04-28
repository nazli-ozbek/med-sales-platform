import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai
from database import get_procedure_by_name
from negotiation.session import NegotiationSession
from state_detector import detect_state

# Ortam değişkenlerini yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# Model ayarları
EMBED_MODEL = "models/embedding-001"
CHAT_MODEL = "gemini-2.0-flash"
INDEX_FOLDER = "indexes/"

# Prosedür Anahtar Kelime Eşlemesi
PROCEDURE_KEYWORDS = {
    "rhinoplasty": ["rhinoplasty", "burun", "burun estetiği"],
    "hair_transplant": ["hair transplant", "saç", "saç ekimi"],
    "liposuction": ["liposuction", "yağ", "yağ aldırma"],
    "dental_implant": ["implant", "diş", "diş implantı"],
}

DEFAULT_PROCEDURE = "rhinoplasty"

def clean_input(text: str) -> str:
    return text.encode("utf-8", "ignore").decode("utf-8", "ignore")

def find_procedure_in_query(user_message: str) -> str:
    user_message = user_message.lower()
    for procedure, keywords in PROCEDURE_KEYWORDS.items():
        for keyword in keywords:
            if keyword in user_message:
                return procedure
    return DEFAULT_PROCEDURE

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

def generate_answer(model_name, question, context):
    model = genai.GenerativeModel(model_name)

    sys_prompt = (
        "You are a smart and concise medical assistant.\n"
        "Answer the user's specific question based primarily on the provided context.\n"
        "If the context contains the necessary information, answer precisely using it.\n"
        "If the context does not explicitly contain the answer but the information can be reasonably inferred, you may generate a basic informative answer.\n"
        "If the topic is entirely unrelated to the context, say 'Not enough information.'\n"
        "Be brief, clear, and to the point."
    )

    prompt = (
        f"{sys_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=256,
        )
    )
    return resp.text.strip()

# ---------------- Ana Program ----------------
def main():
    print("=== Çoklu Prosedür Destekli RAG Chatbot ===")
    print("Soru sorabilirsiniz. Çıkmak için 'quit' yazın.\n")

    last_session = None
    last_procedure = DEFAULT_PROCEDURE
    faiss_index, chunks = load_faiss_index_and_chunks(last_procedure)
    procedure_info = get_procedure_by_name(last_procedure)
    session = NegotiationSession(procedure_info)

    while True:
        raw_query = input("Soru: ").strip()

        if not raw_query:
            continue

        user_query = clean_input(raw_query)

        if user_query.lower() in ["quit", "exit"]:
            print("Görüşmek üzere!")
            break

        detected_procedure = find_procedure_in_query(user_query)
        print(f"[DEBUG] Detected Procedure: {detected_procedure}")

        if detected_procedure != last_procedure:
            faiss_index, chunks = load_faiss_index_and_chunks(detected_procedure)
            procedure_info = get_procedure_by_name(detected_procedure)
            session = NegotiationSession(procedure_info)
            last_procedure = detected_procedure

        detected_state = detect_state(user_query)
        print(f"[DEBUG] Detected State: {detected_state}")

        if detected_state == "ASK_PRICE":
            print(f"The base price for {detected_procedure} is {procedure_info['base_price']}₺.")
            continue

        elif detected_state == "NEGOTIATE":
            negotiation_response = session.respond(user_query)
            print("Cevap:", negotiation_response)
            continue

        elif detected_state == "ACCEPT":
            negotiation_response = session.respond(user_query)
            print("Cevap:", negotiation_response)
            continue

        else:  # ASK_INFO veya default
            query_vec = embed_query(user_query)
            best_chunks = find_relevant_chunks(faiss_index, chunks, query_vec, top_k=2)
            context_text = "\n\n".join(best_chunks)
            answer = generate_answer(CHAT_MODEL, user_query, context_text)

            print("\nCevap:\n", answer, "\n")

if __name__ == "__main__":
    main()
