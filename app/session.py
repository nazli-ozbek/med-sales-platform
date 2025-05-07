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
        self.doctor = None  # yeni alan

    def set_doctor(self, doctor_info):
        self.doctor = doctor_info

    def respond(self, message):
        self.history.append({"user": message})
        current_polarity = TextBlob(message).sentiment.polarity
        self.polarity_history.append(current_polarity)
        avg_polarity = sum(self.polarity_history) / len(self.polarity_history)

        prompt = self._build_prompt(message, current_polarity, avg_polarity)
        response = self._ask_llm(prompt)

        self.history.append({"llm": response})
        return response

    def _build_prompt(self, user_message, current_polarity, avg_polarity):
        history_lines = ""
        for h in self.history:
            if "user" in h:
                history_lines += f"User: {h['user']}\n"
            elif "llm" in h:
                history_lines += f"Agent: {h['llm']}\n"

        return f'''
        You are an AI sales agent negotiating ONLY the price of a medical procedure.

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