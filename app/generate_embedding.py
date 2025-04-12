import os
import textwrap
import faiss
import numpy as np
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Ortam değişkenlerini yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Gemini API'yi başlat
genai.configure(api_key=GEMINI_API_KEY)

EMBED_MODEL = "models/embedding-001"  # Örnek bir Gemini embedding modeli

FILE_PATH = "docs/procedures/rhinoplasty_info.txt"

def main():
    with open(FILE_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    # 1) Metni chunk'lara böl
    chunks = textwrap.wrap(text, 300)

    print("→ Gemini üzerinden embedding alınıyor...")
    vectors = []
    for chunk in chunks:
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=chunk,
            task_type="retrieval_document"
        )
        # Sonuç bir dict, "embedding" anahtarını okuyacağız
        emb = result["embedding"]  # Tek chunk = tek embedding
        vectors.append(emb)

    # 2) NumPy array'e çevir
    vectors_np = np.array(vectors).astype("float32")

    # 3) FAISS index
    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)

    print("→ Embedding tamamlandı. FAISS index oluşturuldu.")

    # 4) Index'i diske kaydet
    faiss.write_index(index, "rhinoplasty.index")
    print("→ FAISS index dosyası kaydedildi: rhinoplasty.index")

    # 5) Chunk'ları JSON olarak saklayalım
    with open("rhinoplasty_chunks.json", "w", encoding="utf-8") as jf:
        json.dump(chunks, jf, ensure_ascii=False, indent=2)
    print("→ Chunk metinleri kaydedildi: rhinoplasty_chunks.json")


if __name__ == "__main__":
    main()
