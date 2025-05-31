import google.generativeai as genai
import json

class QuestionnaireManager:
    def __init__(self):
        self.questions = [
            "1. What is your full name?",
            "2. What is your age?",
            "3. Do you have any allergies?",
            "4. What is your expected date for the surgery?",
            "5. Do you have any contagious diseases (e.g., Hepatitis B, Hepatitis C, HIV)?",
            "6. Are you taking any medications or have any ongoing health problems?",
            "7. What is your height and weight?",
            "8. Have you had any previous surgeries?"
        ]
        self.answers = {}
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")

    def is_complete(self):
        return len(self.answers) == len(self.questions)

    def get_all_answers(self):
        return self.answers

    def get_all_questions(self):
        return self.questions

    def get_unanswered_questions(self):
        return [q for i, q in enumerate(self.questions, 1) if f"Q{i}" not in self.answers]

    def process_bulk_response(self, user_response):
        prompt = (
                "You are an AI assistant helping fill out a medical intake form. Below are 8 questions "
                "and a user response. Extract all the answers you can, and return only a valid JSON object "
                "mapping question numbers (Q1–Q8) to answers. If any question is not answered, simply skip it.\n\n"
                "QUESTIONS:\n" + "\n".join(self.questions) +
                f"\n\nUSER RESPONSE:\n{user_response}\n\n"
                "Return result in this format ONLY (strict JSON):\n"
                "{\n  \"Q1\": {\"question\": \"...\", \"answer\": \"...\"},\n  \"Q2\": {...},\n  ...\n}"
        )

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()

            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]

            parsed = json.loads(response_text)

            for q_num, q_obj in parsed.items():
                if q_num not in self.answers:
                    self.answers[q_num] = q_obj

            import os
            # Q1 cevabını dosya adı olarak kullan
            user_name = self.answers.get("Q1", {}).get("answer", "unknown_user")
            filename = user_name.strip().replace(" ", "_").replace("/", "_").lower() + "_questionnaire.json"

            # Dosyaya yaz
            os.makedirs("saved_forms", exist_ok=True)
            with open(f"saved_forms/{filename}", "w", encoding="utf-8") as f:
                json.dump(self.answers, f, indent=2, ensure_ascii=False)


            return True

        except Exception as e:
            print("[ERROR parsing LLM JSON]", e)
            return False
