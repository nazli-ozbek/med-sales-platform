import google.generativeai as genai

class SummarizerAgent:
    def __init__(self, model_name="gemini-2.0-flash"):
        self.model = genai.GenerativeModel(model_name)
        self.summary = ""

    def update_summary(self, chat_history):
        history_text = "\n".join(chat_history)
        prompt = (
            "You are a summarizer for a medical chatbot. Your job is to iteratively improve the previous summary "
            "based on the latest chat updates.\n\n"
            f"Previous Summary:\n{self.summary}\n\n"
            f"New Chat:\n{history_text}\n\n"
            "Updated Summary:"
        )
        response = self.model.generate_content(prompt)
        self.summary = response.text.strip()

    def get_summary(self):
        return self.summary
