# CURA: Medical Assistant Chatbot

CURA is an **AI-powered medical assistant chatbot** designed to provide contextual, emotionally intelligent, and personalized support for patients seeking cosmetic medical procedures such as rhinoplasty, liposuction, and dental implants.  

Unlike rule-based FAQ bots, CURA integrates **Large Language Models (LLMs)**, **semantic search**, and **sentiment-aware negotiation logic** to deliver natural, empathetic, and safe multi-turn conversations.

---

## ✨ Features

- 🧠 **Intelligent Conversations** – Context-aware responses powered by Google Gemini.
- ❤️ **Emotionally Aware** – Sentiment analysis influences tone and negotiation strategies.
- 🔍 **Semantic Search** – Uses Pinecone to retrieve procedure-specific medical information.
- 📋 **Medical Intake Form** – Secure, structured questionnaire ensures safety and personalization.
- 🤝 **Price Negotiation** – Human-like negotiation logic with ethical constraints.
- 💻 **Modern Web UI** – Built with React.js, featuring dark/light mode, auto-scroll, and animations.
- 🔒 **Privacy by Design** – No personal data sent to third-party APIs; local storage for form data.

---

## 🏗️ System Architecture

CURA follows a **modular and scalable architecture**:

1. **Frontend (React.js)**  
   - Chat bubble UI with animations (Framer Motion)  
   - Auto-scrolling, input handling, and theme switching  

2. **Backend (FastAPI)**  
   - Conversation flow & state management  
   - Sentiment analysis with TextBlob  
   - Integration with Google Gemini for LLM responses  
   - PostgreSQL for doctors, procedures, and negotiation data  

3. **Semantic Retrieval (Pinecone)**  
   - Stores embeddings of medical procedure documents  
   - Contextual retrieval for accurate and grounded responses  

---

## 🛠️ Technologies

### Backend
- [Python](https://www.python.org/) & [FastAPI](https://fastapi.tiangolo.com/)  
- [Google Gemini](https://deepmind.google/technologies/gemini/) (LLM & embeddings)  
- [Pinecone](https://www.pinecone.io/) – Vector database  
- [PostgreSQL](https://www.postgresql.org/) + Psycopg2  
- [TextBlob](https://textblob.readthedocs.io/) – Sentiment analysis  
- [Pydantic](https://docs.pydantic.dev/) – Data validation  

### Frontend
- [React.js](https://react.dev/) – SPA frontend  
- [Framer Motion](https://www.framer.com/motion/) – Animations  
- Custom theming (Dark/Light mode)  

### Other Tools
- Dotenv for environment management  
- CORS middleware for secure frontend-backend communication  

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+  
- Node.js 18+  
- PostgreSQL  
- Pinecone API key  
- Google Gemini API key  

### Backend Setup
```bash
# Clone the repository
git clone https://github.com/your-username/cura-chatbot.git
cd cura-chatbot/backend

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI server
uvicorn main:app --reload
```


### Frontend Setup
```bash
cd ../frontend

# Install dependencies
npm install

# Start development server
npm run dev

Access the app at: http://localhost:3000
```

## 📊 Evaluation

- ✅ Natural & empathetic responses  
- ✅ Accurate retrieval of medical knowledge  
- ✅ Smooth state transitions and negotiation flow  
- ⚠️ Limitations: dependency on LLM stability, static doctor database, and limited multilingual support  

---

## 🔮 Future Work

- 🌍 Multilingual support (starting with Turkish)  
- 📅 Real-time doctor scheduling & appointment booking  
- 🧾 Long-term user memory across sessions  
- 📊 Richer procedure library with visual aids  
- 🔗 Integration with Electronic Health Record (EHR) systems  

---

## 👩‍💻 Authors

- **Nazlı Çiçek Özbek**
- **Mert Korkmaz**  

---

## 📜 License
This project is for **academic and research purposes**.  
For commercial or clinical use, please ensure compliance with relevant healthcare and data protection regulations.
