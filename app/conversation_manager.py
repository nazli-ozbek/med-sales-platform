# conversation_manager.py

class ConversationManager:
    def __init__(self):
        self.current_state = None

        # Örnek geçiş tablosu
        self.transition_map = {
            "LATENT_INTEREST": "ASK_INFO",
            "ASK_INFO": "ASK_RISKS",
            "ASK_RISKS": "ASK_RECOVERY",
            "ASK_RECOVERY": "END",
            "NEGOTIATE": "END",
            "ACCEPT": "END",
            "ASK_PRICE": "NEGOTIATE",
            "ASK_ALTERNATIVES": "END",
            "ESCALATE": "HUMAN",
        }

    def update_state(self, new_state):
        """
        Geçerli state'i günceller
        """
        self.current_state = new_state

    def get_next_state(self):
        """
        Geçerli state'e göre bir sonraki state'i döner
        """
        return self.transition_map.get(self.current_state, "ASK_INFO")

    def is_end(self):
        """
        Sohbetin sonuna gelinip gelinmediğini kontrol eder
        """
        return self.get_next_state() == "END"
