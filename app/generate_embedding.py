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
genai.configure(api_key=GEMINI_API_KEY)

# Embedding ayarları
EMBED_MODEL = "models/embedding-001"
PROCEDURE_FOLDER = "docs/procedures/"
INDEX_FOLDER = "indexes/"

# Klasörleri kontrol et
os.makedirs(INDEX_FOLDER, exist_ok=True)

def generate_embeddings_for_file(filepath):
    """Belirli bir dosya için FAISS index ve chunks.json oluşturur"""
    filename = os.path.basename(filepath)
    procedure_name = os.path.splitext(filename)[0]

    # Dosya içeriğini oku
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    # Metni chunk'lara böl
    chunks = textwrap.wrap(text, width=150)

    # Her chunk için embedding al
    vectors = []
    for chunk in chunks:
        response = genai.embed_content(
            model=EMBED_MODEL,
            content=chunk,
            task_type="retrieval_document"
        )
        emb = response["embedding"]
        vectors.append(emb)

    vectors_np = np.array(vectors).astype("float32")

    # FAISS index oluştur
    index = faiss.IndexFlatL2(vectors_np.shape[1])
    index.add(vectors_np)

    # FAISS index kaydet
    index_path = os.path.join(INDEX_FOLDER, f"{procedure_name}.index")
    faiss.write_index(index, index_path)

    # Chunk'ları JSON olarak kaydet
    chunks_path = os.path.join(INDEX_FOLDER, f"{procedure_name}_chunks.json")
    with open(chunks_path, "w", encoding="utf-8") as jf:
        json.dump(chunks, jf, ensure_ascii=False, indent=2)

    print(f"✅ {procedure_name} için embedding tamamlandı ve kayıt edildi.")

def main():
    """Tüm prosedür dosyalarını bulur ve embedding üretir"""
    files = [os.path.join(PROCEDURE_FOLDER, f) for f in os.listdir(PROCEDURE_FOLDER) if f.endswith(".txt")]

    if not files:
        print("❗ Prosedür dosyası bulunamadı!")
        return

    for filepath in files:
        generate_embeddings_for_file(filepath)

    print("\n🎉 Tüm prosedürler için embedding işlemi tamamlandı!")

if __name__ == "__main__":
    main()