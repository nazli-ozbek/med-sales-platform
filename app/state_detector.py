import google.generativeai as genai

def detect_state(user_message, last_agent_message=None) -> str:

    """
    Kullanıcının mesajından niyet (intent) tespiti yapar.
    Genişletilmiş sınıflarla semantic sınıflandırma yapar.
    """

    model = genai.GenerativeModel("gemini-2.0-flash")

    system_prompt = (
        "You are an AI assistant classifying user intent regarding medical procedures.\n\n"
        "Classify the user's message into one of the following intents:\n"
        "- LATENT_INTEREST: User expresses personal dissatisfaction (e.g. 'I don't like my nose') without directly asking for info.\n"
        "- ASK_INFO: User requests general details (how it's done, what it is, etc.)\n"
        "- ASK_PRICE: User asks about cost or price.\n"
        "- NEGOTIATE: User makes a price offer or asks for discount.\n"
        "- ACCEPT_PRICE: User accepts the offered price, but has not yet confirmed the procedure.\n"
        "- ASK_RISKS: User wants to know about risks, complications, or side effects.\n"
        "- ASK_RECOVERY: User inquires about healing, downtime, or recovery.\n"
        "- ASK_ALTERNATIVES: User looks for other supporting treatments or alternatives.\n"
        "- ESCALATE: Complex issue that needs human support (e.g. legality, emotional distress).\n\n"
        "IMPORTANT:\n"
        "- Respond ONLY with one of the labels above.\n"
        "- Do NOT explain.\n"
        "- If unsure, default to 'ASK_INFO'.\n"
        "-If the context involves price, treat it as ASK_PRICE by default.\n"
        "-If the user replies with yes, sure, or okay, check context to disambiguate.\n"
        "-Only assign ACCEPT_PRICE if it is clear that the user confirms they want to proceed with the procedure by checking the context.\n."
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
