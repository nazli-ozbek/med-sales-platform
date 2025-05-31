import os
import textwrap
import numpy as np
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone, ServerlessSpec

# Ortam değişkenlerini yükle
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX", "procedure-embeddings")

# Sabit ayarlar
EMBED_MODEL = "models/embedding-001"
PROCEDURE_FOLDER = "docs/procedures/"

# Gemini API yapılandırması
genai.configure(api_key=GEMINI_API_KEY)

# ✅ Pinecone nesnesi oluştur
pc = Pinecone(api_key=PINECONE_API_KEY)
# index silme
pc.delete_index("procedure-embeddings")

# yeniden oluşturmak için generate_embedding_pinecone.py script’ini tekrar çalıştır

# ✅ Eğer index yoksa oluştur
if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=768,  # Gemini'nin embedding boyutu
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"  # İstersen değiştirebilirim
        )
    )

# ✅ Index objesi al
index = pc.Index(INDEX_NAME)

def generate_embeddings_for_file(filepath):
    """Belirli bir prosedür dosyası için embedding üretip Pinecone'a yükler."""
    filename = os.path.basename(filepath)
    procedure_name = os.path.splitext(filename)[0]

    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

    chunks = textwrap.wrap(text, width=150)

    pinecone_entries = []

    for i, chunk in enumerate(chunks):
        try:
            response = genai.embed_content(
                model=EMBED_MODEL,
                content=chunk,
                task_type="retrieval_document"
            )
            emb = response["embedding"]
            pinecone_entries.append({
                "id": f"{procedure_name}-{i}",
                "values": emb,
                "metadata": {
                    "text": chunk,
                    "procedure": procedure_name
                }
            })
        except Exception as e:
            print(f"[ERROR] Chunk {i} embedding failed: {e}")

    if pinecone_entries:
        index.upsert(vectors=pinecone_entries)
        print(f"✅ {procedure_name} için {len(pinecone_entries)} embedding yüklendi.")
    else:
        print(f"⚠️  {procedure_name} için embedding oluşturulamadı.")

def main():
    if not os.path.exists(PROCEDURE_FOLDER):
        print(f"❗ {PROCEDURE_FOLDER} klasörü bulunamadı!")
        return

    files = [os.path.join(PROCEDURE_FOLDER, f) for f in os.listdir(PROCEDURE_FOLDER) if f.endswith(".txt")]

    if not files:
        print("❗ Prosedür dosyası bulunamadı!")
        return

    for filepath in files:
        generate_embeddings_for_file(filepath)

    print("\n🎉 Tüm prosedürler başarıyla Pinecone'a yüklendi!")

if __name__ == "__main__":
    main()
