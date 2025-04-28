import google.generativeai as genai

def detect_state(user_message):
    """
    Kullanıcının mesajından niyet (intent) tespiti yapar.
    LLM semantic anlam çıkararak sınıflandırır: ASK_INFO, ASK_PRICE, NEGOTIATE, ACCEPT
    """

    model = genai.GenerativeModel("gemini-2.0-flash")

    system_prompt = (
        "You are an AI assistant helping to classify the user's intent regarding a medical procedure.\n\n"
        "Given the user's full message, determine the intent based on semantic meaning, NOT keywords.\n\n"
        "Possible intents:\n"
        "- ASK_INFO: User wants general information about the procedure.\n"
        "- ASK_PRICE: User asks about the price or cost.\n"
        "- NEGOTIATE: User tries to negotiate or offers a counterprice.\n"
        "- ACCEPT: User accepts an offer, agrees, approves, or consents to a deal.\n\n"
        "IMPORTANT:\n"
        "- If the user seems to agree, approve, confirm or accept a deal — classify as ACCEPT even if they use informal expressions like 'okay', 'sounds good', 'sure', 'olur', 'tamam', 'peki', 'ok', etc.\n"
        "- If the user talks about money offers ('2500 olur mu?', 'biraz indirim yapar mısınız?') — classify as NEGOTIATE.\n"
        "- If asking about the cost directly ('fiyat nedir?', 'ne kadar?') — classify as ASK_PRICE.\n"
        "- If asking about general details, risks, recovery time — classify as ASK_INFO.\n\n"
        "If uncertain, default to ASK_INFO.\n\n"
        "Respond ONLY with one of these labels: ASK_INFO, ASK_PRICE, NEGOTIATE, ACCEPT.\n"
        "Respond with ONLY the label text, nothing else."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"User Message: {user_message}\n"
        f"Answer:"
    )

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=10,
        )
    )

    predicted_state = response.text.strip()
    return predicted_state
