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
        self.current_index = 0

    def get_next_question(self):
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    def answer_current_question(self, user_response):
        if self.current_index < len(self.questions):
            q_text = self.questions[self.current_index]
            q_key = f"Q{self.current_index + 1}"
            self.answers[q_key] = {
                "question": q_text,
                "answer": user_response
            }
            self.current_index += 1

    def is_complete(self):
        return self.current_index >= len(self.questions)

    def get_all_answers(self):
        return self.answers