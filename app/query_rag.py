import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
import google.generativeai as genai

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

def generate_answer(chat_model_name, question, context):
    """
    Gemini modelini kullanarak, context'e dayalı yanıt oluşturur.
    """
    # 1) Model nesnesi oluştur
    model = genai.GenerativeModel(chat_model_name)

    # 2) Prompt'u hazırla
    system_prompt = (
        "You are a helpful and informative medical assistant. "
        "You answer questions using ONLY the provided context. "
        "If the answer is not in the context, say 'Not enough information'."
    )
    full_prompt = (
        f"{system_prompt}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    # 3) Yanıtı üret
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

    while True:
        user_query = input("Soru: ").strip()
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
        answer = generate_answer(CHAT_MODEL, user_query, context_text)
        print("\nCevap:\n", answer, "\n")

if __name__ == "__main__":
    main()
