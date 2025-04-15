import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai
from textblob import TextBlob
from database import get_procedure_by_name
import re


# RAG adımları:
# 1) Soru al (kullanıcı input)
# 2) Soruya embedding bul (Gemini)
# 3) FAISS search ile en ilgili chunkları bul
# 4) Gemini'ye "Bu chunk'a göre cevap ver" diye prompt oluştur
# 5) Yanıtı ekrana bas

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini ayarları
genai.configure(api_key=GEMINI_API_KEY)
EMBED_MODEL = "models/embedding-001"   # Aynı embedding modeli
CHAT_MODEL  = "gemini-2.0-flash"       # Örnek bir chat modeli

def load_faiss_index(index_path="rhinoplasty.index"):
    """Diskten FAISS index ve chunk verisini yükle"""
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index dosyası bulunamadı: {index_path}")

    # FAISS index'i oku
    index = faiss.read_index(index_path)

    # Chunk metinlerini de JSON'dan al
    with open("rhinoplasty_chunks.json", "r", encoding="utf-8") as jf:
        chunks = json.load(jf)

    return index, chunks

def clean_input(text):
    return text.encode('utf-8', 'ignore').decode('utf-8')

def extract_offer_from_text(text):
    matches = re.findall(r"\d+", text)
    if matches:
        return int(matches[0])
    return None

def negotiate_price(offer, procedure, last_offer=None):
    base = procedure["base_price"]
    min_price = procedure["bargain_min"]

    if offer >= base or last_offer is not None and offer >= last_offer:
        return "That's great! We can proceed with the treatment.", None
    elif min_price <= offer < base:
        counter = (offer + base) // 2
        return f"This price is a bit low. I can offer you a special deal at ${counter}.", counter
    else:
        return f"Sorry, this offer is too low.", None



def embed_query(query):
    """Sorgu cümlesini Gemini ile vektöre dönüştür"""
    response = genai.embed_content(
        model=EMBED_MODEL,
        content=query,
        task_type="retrieval_document"
    )
    q_vector = response["embedding"]  # Tek embedding
    # FAISS'e arama yapabilmek için [1 x dimension] boyutuna getirelim
    q_vec_np = np.array(q_vector, dtype="float32").reshape(1, -1)
    return q_vec_np

def find_relevant_chunks(faiss_index, chunks, query_vector, top_k=2):
    """FAISS araması: en yakın top_k paragrafı bul"""
    distances, indices = faiss_index.search(query_vector, top_k)
    # indices shape: (1, top_k)
    best_chunks = []
    for i in indices[0]:
        best_chunks.append(chunks[i])
    return best_chunks

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return "positive"
    elif polarity < -0.1:
        return "negative"
    else:
        return "neutral"

def generate_answer(chat_model_name, question, context, sentiment="neutral"):
    model = genai.GenerativeModel(chat_model_name)

    if sentiment == "negative":
        system_prompt = (
            "You are a caring and reassuring medical assistant. "
            "The user seems concerned or negative, so try to comfort and persuade them. "
            "You answer ONLY using the provided context. If the answer is not in the context, say 'Not enough information'."
        )
    elif sentiment == "positive":
        system_prompt = (
            "You are a cheerful and helpful medical assistant. "
            "The user seems happy, so continue the flow positively and show enthusiasm. "
            "You answer ONLY using the provided context. If the answer is not in the context, say 'Not enough information'."
        )
    else:
        system_prompt = (
            "You are a helpful and informative medical assistant. "
            "You answer ONLY using the provided context. If the answer is not in the context, say 'Not enough information'."
        )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=256,
        )
    )

    return response.text.strip()



def main():
    print("=== RAG Query Demo ===")
    print("Bu program FAISS index'i ve Gemini'yi kullanarak soru cevaplıyor.")
    print("Çıkmak için 'quit' yaz.\n")

    # 1) FAISS index ve chunk verisini yükle
    index, chunks = load_faiss_index("rhinoplasty.index")
    last_counter_offer = None
    while True:
        user_query_raw = input("Soru: ").strip()
        user_query = clean_input(user_query_raw)
        sentiment = analyze_sentiment(user_query)

        procedure_name = "rhinoplasty"  # Şimdilik manuel, ileride otomatikleştirilir
        procedure = get_procedure_by_name(procedure_name)
        offer = extract_offer_from_text(user_query)
        if offer is not None and procedure:
            negotiation_response, new_counter = negotiate_price(offer, procedure, last_counter_offer)
            print("Cevap:", negotiation_response)
            last_counter_offer = new_counter
            continue

        print(f"[DEBUG] Kullanıcı duygu analizi sonucu: {sentiment}")

        if user_query.lower() in ["quit", "exit"]:
            print("Görüşmek üzere!")
            break

        # 2) Query embedding
        query_vec = embed_query(user_query)

        # 3) FAISS ile en ilgili 2 chunk
        best_chunks = find_relevant_chunks(index, chunks, query_vec, top_k=2)

        # 4) Chunk metinlerini birleştir
        context_text = "\n\n".join(best_chunks)

        # 5) Yanıtı al
        answer = generate_answer(CHAT_MODEL, user_query, context_text, sentiment)
        print("\nCevap:\n", answer, "\n")

if __name__ == "__main__":
    main()
