class NegotiationSession:
    def __init__(self, procedure):
        self.base = procedure["base_price"]
        self.min_price = procedure["bargain_min"]
        self.max_price = procedure["bargain_max"]
        self.last_counter_offer = None

    def respond(self, message):
        offer = self.extract_offer(message)

        if offer:
            if offer >= self.base:
                return "That's a great offer! We can proceed with the treatment."
            elif self.min_price <= offer < self.base:
                self.last_counter_offer = (offer + self.base) // 2
                return f"This is a bit low. I can offer you a special deal at {self.last_counter_offer}₺."
            else:
                return "Sorry, your offer is too low for this procedure."

        if "accept" in message.lower() or "kabul" in message.lower():
            if self.last_counter_offer:
                return f"Great! You accepted the counter offer of {self.last_counter_offer}₺."
            else:
                return "We haven't agreed on a final price yet. Let's discuss a deal first."

        return "Could you please make an offer?"

    def extract_offer(self, text):
        import re
        matches = re.findall(r"\d+", text)
        if matches:
            return int(matches[0])
        return None
