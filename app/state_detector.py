import google.generativeai as genai

def detect_state(user_message, last_agent_message=None) -> str:

    """
    Kullanıcının mesajından niyet (intent) tespiti yapar.
    Genişletilmiş sınıflarla semantic sınıflandırma yapar.
    """

    model = genai.GenerativeModel("gemini-2.0-flash-lite")

    system_prompt = (
        "You are an AI assistant classifying user intent regarding medical procedures.\n\n"
        "Classify the user's message into one of the following intents as current state:\n\n"

        "- QUESTIONNAIRE: The user is in the early phase of the conversation and must complete a short medical intake form. "
        "This includes messages where the user is answering personal medical questions such as name, age, allergies, surgery date, etc.\n"
        "Examples: 'Hi', 'My name is Sarah', 'I am 29', 'No allergies', 'I had no previous surgeries'\n\n"
        "- LATENT_INTEREST: The user expresses dissatisfaction with a body part or appearance without directly asking questions. \n"
        "  Example: 'I don’t like my nose.' / 'I wish my teeth looked better.'\n\n"
        "- ASK_INFO: The user wants to know what the procedure is, how it's performed, or general information. \n"
        "  Example: 'How does rhinoplasty work?' / 'Can you tell me what liposuction is?'\n\n"
        "- ASK_PRICE: The user asks about the cost or price of the procedure. \n"
        "  Example: 'How much does it cost?' / 'What’s the price for this treatment?'\n\n"
        "- NEGOTIATE: The user reacts to a price by expressing it is too high, asks for a discount, or offers a counter-price. \n"
        "  Example: 'That’s too much.' / 'Can you do it for 1200?' / 'Is there a cheaper option?'\n\n"
        "- ACCEPT_PRICE: The user clearly accepts the offered price and indicates intent to proceed with the treatment. \n"
        "  Example: 'Okay, I’ll do it.' / 'That works for me.' / 'Let’s go ahead.'\n\n"
        "- ASK_RISKS: The user is asking about risks, complications, or possible negative outcomes. \n"
        "  Example: 'Is it dangerous?' / 'Are there any side effects?'\n\n"
        "- ASK_RECOVERY: The user wants to know about healing time, downtime, or what to expect after the procedure. \n"
        "  Example: 'How long will I need to rest?' / 'When can I go back to work?'\n\n"
        "- ASK_ALTERNATIVES: The user asks about alternative treatments or other ways to achieve similar results. \n"
        "  Example: 'Is there a non-surgical option?' / 'Are there any creams instead of surgery?'\n\n"
        "- ESCALATE: The user expresses strong emotions, anxiety, ethical/legal concerns, or situations that require human attention. \n"
        "  Example: 'I’m really scared and don’t know what to do.' / 'Is this even legal?' / 'I feel very anxious about this.'\n\n"
        "IMPORTANT INSTRUCTIONS:\n"
        "- Respond ONLY with one of the labels above. Do NOT include explanations.\n"
        "- If unsure, default to 'ASK_INFO'.\n"
        "- If the context involves price in any form, and there is no bargaining, classify it as ASK_PRICE.\n"
        "- If the user says things like 'yes', 'okay', or 'sure', use the prior assistant message to decide the intent.\n"
        "- ONLY classify as ACCEPT_PRICE if the user clearly agrees to the price and wants to proceed with the treatment.\n"
    )

    if last_agent_message:
        system_prompt += (
            "\nContext:\n"
            f"The assistant previously said: \"{last_agent_message.strip()}\"\n"
            "Consider the context to disambiguate short or vague user replies.\n"
        )

    full_prompt = f"{system_prompt}\n\nUser Message: {user_message}\nAnswer:"

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.0,
            max_output_tokens=10,
        )
    )

    return response.text.strip().upper()

def detect_procedure(user_message: str) -> str:
    """
    Kullanıcı mesajına göre prosedür adını belirler.
    LLM üzerinden çalışır.
    """

    # Burada model adın sabit olsun
    model = genai.GenerativeModel("gemini-2.0-flash")

    system_prompt = (
        "You are an AI assistant that classifies user messages into medical procedures.\n\n"
        "Possible procedures:\n"
        "- rhinoplasty\n"
        "- hair_transplant\n"
        "- liposuction\n"
        "- dental_implant\n\n"
        "If the user's message clearly relates to one of these, respond ONLY with the procedure name.\n"
        "If you are unsure or if the message is irrelevant, respond ONLY with 'unknown'.\n"
        "Do not explain. Respond only with the label text."
    )

    full_prompt = (
        f"{system_prompt}\n\n"
        f"User Message: {user_message}\n"
        f"Answer:"
    )

    response = model.generate_content(
        full_prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.1,
            max_output_tokens=10,
        )
    )

    return response.text.strip().lower()
