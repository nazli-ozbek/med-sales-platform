
# 🏥 AI-Powered Medical Sales Platform

Bu proje, doktorlar ve hastalar arasında medikal işlemler hakkında çok dilli ve yapay zekâ destekli iletişimi kolaylaştıran bir satış platformudur. 

Platform; LLM tabanlı yanıt üretimi (Gemini), metin üzerinden embedding çıkarımı, FAISS ile bilgi arama, basit sentiment analysis, ve doktor-prosedür ilişkili veritabanı yönetimini içermektedir.

---

## 🚀 Özellikler

- 💬 **LLM destekli Chatbot** (Gemini API)
- 📚 **RAG (Retrieval-Augmented Generation)** ile bilgi temelli yanıt üretimi
- 🧠 **Sentiment Analysis** ile kullanıcının duygusuna göre cevap biçimlendirme
- 📂 **FAISS** ile vektör arama
- ⚙️ **SQLAlchemy** ile veritabanı yönetimi (Doctor, Procedure tabloları)
- 🌐 **FastAPI** ile RESTful backend (ileriki aşamalar için)
- 🔐 `.env` ile gizli anahtar yönetimi

---

## 🔧 Gereksinimler

Python 3.10+ sürümü önerilir.

### 📦 Gerekli Kütüphaneler

```bash
pip install -r requirements.txt
```

### 🧠 NLP Model Kurulumu (TextBlob)

```bash
python -m textblob.download_corpora
```

> macOS kullanıyorsan sertifika hataları için:  
> `/Applications/Python\ 3.x/Install\ Certificates.command` komutunu çalıştırmalısın.

---

## ⚙️ Çalıştırma

### 1. Ortam Değişkenlerini Ayarla

`.env` dosyası oluştur ve içine şunları ekle:

```
GEMINI_API_KEY=your_google_generativeai_key_here
```

### 2. FAISS Index ve Chunk Dosyalarını Oluştur

Örnek dosyalar:

- `rhinoplasty_chunks.json`
- `rhinoplasty.index`

> Bu dosyalar medical prosedür içeriği embedding'lenerek oluşturulmalıdır. Örnek dosyalarla çalışıyorsanız klasörle birlikte gelmiş olabilir.

### 3. Uygulamayı Başlat

```bash
python app/query_rag.py
```

---

## 💬 Kullanım Akışı

1. Kullanıcıdan doğal dilde bir soru alınır (örn: "Bu işlem güvenli mi?")
2. Soru Gemini ile embedding’e dönüştürülür.
3. FAISS vektör veritabanından en alakalı içerikler bulunur.
4. Gemini modeli, sadece bu içeriklere dayanarak cevap üretir.
5. Kullanıcının mesajı aynı zamanda sentiment analizine tabi tutulur.
   - Negatifse: ikna edici ve empatik bir tonla yanıt verilir.
   - Pozitifse: olumlu yanıt pekiştirilir.

---

## 📁 Proje Yapısı

```
med-sales-platform/
│
├── app/
│   ├── query_rag.py           # LLM + RAG + Sentiment Analysis
│   ├── generate_embedding.py  # Embedding ve FAISS index oluşturma
│   └── __init__.py
│
├── rhinoplasty_chunks.json    # Metin chunk verileri
├── rhinoplasty.index          # FAISS index dosyası
├── .env                       # API anahtarları
└── README.md
```

---

## 🧪 Geliştirme Planı

- Türkçe destekli sentiment analiz modeline geçiş
- FastAPI ile chatbot endpoint’i oluşturulması
- Kullanıcıdan randevu alma akışı
- Doktor bazlı satış stratejileri

---

## 🧑‍⚕️ Not

Bu proje yalnızca teknik demo amaçlıdır. Gerçek tıbbi tavsiye veya işlem satışı sağlamaz.
```

---

İstersen bunu `.md` uzantılı dosya olarak da çıkartıp indirebilmen için sana yükleyebilirim. İster misin?