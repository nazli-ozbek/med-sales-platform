import google.generativeai as genai
from textblob import TextBlob

class NegotiationSession:
    def __init__(self, procedure):
        self.base = procedure["base_price"]
        self.min_price = procedure["bargain_min"]
        self.max_price = procedure["bargain_max"]
        self.last_offer = None
        self.history = []
        self.polarity_history = []
        self.model = genai.GenerativeModel("gemini-2.0-flash-lite")
        self.procedure = procedure
        self.doctor = None
        self.session_closed = False
        self.medical_profile = {}

    def set_doctor(self, doctor_info):
        self.doctor = doctor_info

    def set_questionnaire(self, profile_data: dict):
        self.medical_profile = profile_data


    def respond(self, message, detected_state=None):
        self.history.append({"user": message})
        current_polarity = TextBlob(message).sentiment.polarity
        self.polarity_history.append(current_polarity)
        avg_polarity = sum(self.polarity_history) / len(self.polarity_history)

        if detected_state == "ACCEPT_PRICE":
            confirmation_prompt = self.confirm_price()
            self.history.append({"llm": confirmation_prompt})
            return confirmation_prompt

        # Normal pazarlık yanıtı üret
        prompt = self._build_prompt(message, current_polarity, avg_polarity)
        response = self._ask_llm(prompt)

        self.last_offer = self.get_offer_from_llm(response)
        self.history.append({"llm": response})
        return response

    def _build_prompt(self, user_message, current_polarity, avg_polarity):
        history_lines = ""
        for h in self.history:
            if "user" in h:
                history_lines += f"User: {h['user']}\n"
            elif "llm" in h:
                history_lines += f"Agent: {h['llm']}\n"
        profile_text = ""
        print({self.base, self.min_price, self.max_price})
        if self.medical_profile:
            profile_lines = []
            for key in sorted(self.medical_profile):
                question = self.medical_profile[key].get("question", "")
                answer = self.medical_profile[key].get("answer", "")
                profile_lines.append(f"- {question} → {answer}")
            profile_text = "\n\nThe user has shared the following medical profile:\n" + "\n".join(profile_lines)

        return f'''
        You are an AI sales agent negotiating ONLY the price of a medical procedure.

        {profile_text}
        The procedure details:
        - Base Price: {self.base}₺
        - Minimum Acceptable Price: {self.min_price}₺
        - Maximum Allowable Price: {self.max_price}₺

        RULES YOU MUST FOLLOW:
        - Only negotiate on the procedure price itself.
        - DO NOT offer additional packages, upgrades, services, or products unless the user specifically asks.
        - NEVER propose a price higher than the Base Price unless the user explicitly offers more.
        - Stay within the negotiation range [{self.min_price}₺ - {self.max_price}₺].
        - If the user's offer is below the Minimum Acceptable Price, decline politely and counter-offer with a fair price.
        - Always stay polite, realistic, and emotionally aware based on the user's emotional tone.
        - Once an offer is made, NEVER increase your next offer beyond that amount.
        - All future offers must either decrease or remain the same.
        - If the user has a medical condition that increases risk (e.g., HIV, heart condition), you may consider offering higher price.
        - NEVER mention the user's medical profile, conditions, or answers explicitly in your message.
        - Use the medical profile only to guide your pricing strategy internally.
        - Your response must appear general and neutral. The user should never know that their profile influenced the price.
        
        Sentiment analysis:
        - Current Message Polarity: {current_polarity:.2f}
        - Average Conversation Polarity: {avg_polarity:.2f}
        - Positive polarity (+1.0): Excited, Trusting.
        - Negative polarity (-1.0): Angry, Doubtful.
        - Neutral (0.0): Uncertain, Neutral.
        Price Strategy:
        - For your first offer, decide on a fair starting price based on the user’s average emotional tone. 
        - The first offer should be between the base price ({self.base}) and the maximum allowed price ({self.max_price})
        - Do not mention the base price, just start the negotiation using the first offer
        - Do NOT mention these strategies or any reasoning to the user.
        - Be adaptive: if the user seems more positive, you may start with a slightly higher price.
        - If the user seems skeptical or negative, consider starting with a more cautious offer.
        - Make sure the price is within the allowed range.
        - Start the negotiation by using this strategy for your initial offer.
        - After your first offer, any future counter-offers MUST be equal to or lower than your previous one.
        - Never increase your price during the negotiation.
        
        If the user accepts the offer (e.g., says “I accept”, “deal”, “okay”, “kabul”, etc.):
        - Respond with an enthusiastic confirmation.
        - Say you will inform the doctor and handle the next steps.
        - Thank the user warmly.
        - End the conversation in a supportive, positive tone.
        - Do not upsell, reopen negotiation, or ask additional questions.
        - Keep the final message short and sincere.
        
        Here is the conversation history so far:
        {history_lines}

        Respond briefly and persuasively based on the conversation context.

        User: {user_message}
        Agent:
        '''.strip()

    def _ask_llm(self, prompt):
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.5,   # Daha az serbestlik
                max_output_tokens=200,  # Fazla uzamasın
            )
        )
        return response.text.strip()

    def confirm_price(self):
        """
        Fiyat anlaşması sonrası kullanıcıdan son onay alacak mesajı üretir
        """
        if not self.last_offer:
            raise ValueError("Henüz teklif yapılmamış.")

        prompt = f"""
        We’ve just agreed on a final price of {self.last_offer}₺.

        Please ask the user for final confirmation. 
        Mention that this decision is irreversible and the doctor will be informed immediately after confirmation. 
        Be warm, supportive, and professional.
        End your message with: "Do you confirm this agreement?"
        """.strip()

        response = self._ask_llm(prompt)
        return response

    def get_offer_from_llm(self, response_text):
        """
        LLM'e tüm geçmişi vererek fiyat teklifini çıkarır.
        Daha bağlamsal ve doğru çalışır.
        """
        # Chat geçmişini derle
        history_lines = ""
        for h in self.history:
            if "user" in h:
                history_lines += f"User: {h['user']}\n"
            elif "llm" in h:
                history_lines += f"Assistant: {h['llm']}\n"

        # LLM'e fiyatı sor
        prompt = (
            "You are a parser that extracts the final price offered by the assistant (in Turkish Lira).\n"
            "Given the entire conversation history, return only the latest proposed price in **numeric form** (integer).\n\n"
            "CONVERSATION HISTORY:\n"
            f"{history_lines}\n"
            f"Current assistant message:\n\"{response_text}\"\n\n"
            "Output example: 1450\n"
            "Do NOT include '₺', 'TL', or any text — only the number."
        )

        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=10,
            )
        )

        try:
            return int(response.text.strip())
        except ValueError:
            return None

