"""
========================================================
 File: src/gemini_analyzer.py
 PURPOSE:
   This file sends prescription data to Google Gemini AI
   and returns two things:
     1. Structured JSON (patient name, medicines, dosages, alternatives)
     2. Conversational answers to follow-up questions

   WHAT IS GEMINI?
   Gemini is Google's large language model (LLM) — similar to ChatGPT.
   It is MULTIMODAL, meaning it can process both text AND images in one call.
   We use the "gemini-3.1-flash-lite-preview" variant — fast and cheap.

 HOW IT FITS IN THE PROJECT:
   app.py calls analyze_prescription(image, ocr_text, api_key, region)
   ↓
   gemini_analyzer.py sends image + OCR text to Gemini API
   ↓
   Gemini corrects OCR errors (using image as ground truth) and
   extracts structure: patient, doctor, medicines, doses, alternatives
   ↓
   Returns a Python dict that app.py renders as medicine cards

   app.py also calls chat_about_prescription() for the Q&A tab.
   ↓
   gemini_analyzer.py sends the full conversation history + prescription
   context to Gemini for multi-turn chat responses.

 KEY DESIGN CHOICE — TWO-MODEL PIPELINE:
   Donut OCR  → fast local model, reads raw text from image
   Gemini     → cloud AI, corrects/structures text AND knows drug brands
   Together they are more accurate than either model alone.
========================================================
"""

import json    # json.loads() parses JSON strings into Python dicts
               # json.dumps() converts Python dicts to JSON strings

import re      # re (regular expressions) — used to clean up Gemini's response
               # Gemini sometimes wraps JSON in ```json ... ``` code fences
               # re.sub() removes those fences before parsing

from PIL import Image   # PIL Image — type annotation for function signatures

from google import genai
# google-generativeai is Google's official Python SDK for Gemini API.
# genai.Client() creates an authenticated client using your API key.
# client.models.generate_content() sends a prompt and returns a response.


# ─────────────────────────────────────────────────────
# SECTION 1: PROMPT TEMPLATES
# ─────────────────────────────────────────────────────

# CONCEPT: A "prompt" is the instruction text we send to the AI.
# The quality of the prompt determines the quality of the output.
# We use Python's .format() method to inject dynamic values
# (like ocr_text, region, n_alts) into the template at runtime.

_EXTRACTION_PROMPT = """
You are an expert medical prescription analyst.
You will receive:
  (a) An image of a handwritten or printed medical prescription.
  (b) OCR text extracted by an AI model (may contain errors or be empty).

Your tasks:
1. Use BOTH the image and the OCR text to accurately extract prescription data.
2. Correct any OCR errors by referring to the actual image.
3. For every medicine, identify its generic name and list {n_alts} common
   alternative brand names available in {region}.
4. Return ONLY a valid JSON object — no markdown fences, no extra text.

OCR Text:
\"\"\"
{ocr_text}
\"\"\"

Return this exact JSON schema:
{{
  "header": {{
    "patient_name": "string or 'Not readable'",
    "age": "string or 'Not readable'",
    "date": "string or 'Not readable'",
    "doctor_name": "string or 'Not readable'",
    "clinic": "string or 'Not readable'",
    "diagnosis": "string or 'Not specified'"
  }},
  "medicines": [
    {{
      "name": "Brand or generic name as written",
      "strength": "e.g. 500mg",
      "frequency": "e.g. Twice daily (BD)",
      "duration": "e.g. 5 days",
      "instructions": "e.g. After meals",
      "generic_name": "INN / generic drug name",
      "drug_class": "e.g. Antibiotic, NSAID, Antacid",
      "alternatives": ["Brand A", "Brand B", "Brand C"]
    }}
  ],
  "warnings": [
    "Any warnings, allergies, or special instructions mentioned"
  ],
  "is_prescription": true
}}

Rules:
- If the image is NOT a prescription, set is_prescription to false and
  return empty arrays for medicines and warnings.
- alternatives must be real, commonly available brands in {region}.
- Return ONLY the JSON object.
"""
# WHY {{}} DOUBLE BRACES?
# Python's .format() uses single braces {} as placeholders.
# To include a literal { or } in the output (needed for JSON), we double them: {{ }}.
# So {{}} becomes {} in the final string after .format() is called.

# WHY IS THE SCHEMA IN THE PROMPT?
# LLMs respond to instructions. By showing Gemini the exact JSON structure
# we want, it reliably produces output in that format. Without this,
# Gemini might return prose instead of parseable JSON.

_CHAT_SYSTEM = """
You are a knowledgeable medical information assistant.
You have analysed a medical prescription and extracted the following structured data:

{analysis_json}

Answer the user's questions about this prescription clearly and helpfully.
Guidelines:
- Use simple, plain language.
- If asked about drug interactions, side effects, or dosage, provide accurate general
  information and always recommend consulting a licensed pharmacist or doctor.
- Never recommend stopping or changing prescribed medication.
- Keep answers concise (3–6 sentences unless more detail is needed).
- Always end with: "Please verify with your doctor or pharmacist."
"""
# WHY INJECT THE PRESCRIPTION JSON?
# Gemini has no memory between API calls. By embedding the full analysis
# JSON into the system prompt, Gemini "remembers" what was prescribed
# and can answer specific questions about those medicines.


# ─────────────────────────────────────────────────────
# SECTION 2: analyze_prescription()
# ─────────────────────────────────────────────────────

def analyze_prescription(
    image: Image.Image,   # PIL Image — the prescription photo
    ocr_text: str,        # Raw text from Donut OCR (may be empty or contain errors)
    api_key: str,         # Google Gemini API key from the user's sidebar input
    region: str = "India",            # Country for brand suggestions (default: India)
    n_alternatives: int = 3,          # How many alternative brands to list per medicine
) -> dict:
    """
    Send the prescription image + OCR text to Gemini and return a structured dict.

    This function implements the CROSS-VERIFICATION pattern:
    - Donut OCR reads image → may make mistakes with handwriting
    - Gemini receives BOTH image + OCR text → corrects mistakes using the image
    - Result is much more accurate than either model alone

    Returns:
        A Python dict matching the JSON schema in _EXTRACTION_PROMPT.
        Example keys: data["header"], data["medicines"], data["warnings"]
    """

    # Create an authenticated Gemini API client.
    # api_key authenticates our requests — without it, the API rejects the call.
    # CONCEPT: An API (Application Programming Interface) is a way for your
    # code to communicate with a remote service (here: Google's AI servers).
    client = genai.Client(api_key=api_key)

    # Build the prompt by injecting dynamic values into the template.
    # .format() replaces {ocr_text}, {region}, {n_alts} with actual values.
    prompt = _EXTRACTION_PROMPT.format(
        ocr_text=ocr_text or "(No OCR text – use the image only)",
        # If ocr_text is empty (""), use a fallback message so Gemini
        # knows to rely only on the image.
        region=region,
        n_alts=n_alternatives,
    )

    # Send both the text prompt AND the image to Gemini in one API call.
    # Gemini processes them together — this is what makes it MULTIMODAL.
    # contents=[prompt, image] — Gemini reads the text instruction first,
    # then examines the image to extract the information.
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        # "Flash" = fast, lightweight model — good for structured extraction.
        # "Preview" = beta/experimental version with latest capabilities.
        contents=[prompt, image],
        # Passing a PIL Image directly — the SDK converts it to base64 internally.
    )

    # Extract the text content from the response.
    # response.candidates  → list of possible responses (usually just 1)
    # [0]                  → take the first (and only) candidate
    # .content.parts       → list of content parts (text, images, etc.)
    # [0].text             → the text string from the first part
    text = response.candidates[0].content.parts[0].text

    # Parse the JSON string into a Python dict and return it.
    # _parse_json() handles cleaning up markdown fences if Gemini adds them.
    return _parse_json(text)


# ─────────────────────────────────────────────────────
# SECTION 3: chat_about_prescription()
# ─────────────────────────────────────────────────────

def chat_about_prescription(
    analysis: dict,        # The structured dict from analyze_prescription()
    question: str,         # The user's current question (e.g., "Can I take with food?")
    history: list[dict],   # Previous conversation turns as list of {"role":..,"content":..}
    api_key: str,          # Google Gemini API key
) -> str:
    """
    Answer a follow-up question about the analysed prescription.

    CONCEPT: Multi-turn conversation
    LLMs like Gemini have no persistent memory between API calls.
    To simulate a conversation, we send the ENTIRE chat history
    with every new message. This way Gemini sees the full context.

    Example history after 2 turns:
    [
      {"role": "user",      "content": "What is Paracetamol used for?"},
      {"role": "assistant", "content": "Paracetamol is a pain reliever..."}
    ]

    Returns:
        A string — Gemini's answer to the user's question.
    """

    client = genai.Client(api_key=api_key)

    # Build the system context — this tells Gemini what role it plays
    # and injects the full prescription JSON so it knows what was prescribed.
    # json.dumps(analysis, indent=2) converts the Python dict to a
    # nicely formatted JSON string that Gemini can read.
    system_ctx = _CHAT_SYSTEM.format(
        analysis_json=json.dumps(analysis, indent=2)
    )

    # Build the full conversation as a list of message dicts.
    # CONCEPT: Gemini's chat API expects alternating "user" and "model" turns.
    # We construct this list manually to include the system context.
    messages = []

    # First turn: inject prescription context as a user message
    messages.append({
        "role": "user",
        "parts": [system_ctx],
        # WHY as user turn? Gemini's API doesn't have a separate "system" role.
        # By putting the context as user + model pair at the start, we
        # effectively set up the AI's "memory" of the prescription.
    })

    # Second turn: model acknowledges it has read the prescription
    messages.append({
        "role": "model",
        "parts": ["Understood. I have reviewed the prescription data and I'm ready to answer questions."],
    })

    # Append all previous conversation turns from session state.
    # This replays the entire chat history so Gemini has full context.
    for msg in history:
        # Streamlit uses "assistant" as the role name, but Gemini expects "model".
        role = "user" if msg["role"] == "user" else "model"
        messages.append({"role": role, "parts": [msg["content"]]})

    # Add the current question as the latest user turn
    messages.append({"role": "user", "parts": [question]})

    # Send the entire conversation history to Gemini.
    # str(messages) converts the list to a string representation.
    # NOTE: This is a simplified approach — a production system would
    # use the proper chat API with multi-turn message objects.
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=str(messages),
    )

    # Extract and return the text response, stripping extra whitespace
    text = response.candidates[0].content.parts[0].text
    return text.strip()


# ─────────────────────────────────────────────────────
# SECTION 4: _parse_json() — PRIVATE HELPER
# ─────────────────────────────────────────────────────

def _parse_json(raw: str) -> dict:
    """
    Parse Gemini's response text into a Python dict.

    WHY IS THIS NEEDED?
    Even though we instruct Gemini to return ONLY JSON, it sometimes:
    - Wraps the JSON in ```json ... ``` markdown code fences
    - Adds a sentence before or after the JSON
    - Returns slightly malformed JSON

    This function handles all those cases robustly.

    Args:
        raw: The raw text string from Gemini's response.

    Returns:
        A Python dict parsed from the JSON content.

    Raises:
        ValueError: If no valid JSON can be extracted after all attempts.
    """

    # Start with the raw response and strip leading/trailing whitespace
    text = raw.strip()

    # ATTEMPT 1: Remove markdown code fences if present.
    # Gemini sometimes returns:
    #   ```json
    #   { "header": {...} }
    #   ```
    # We need to remove the ``` lines to get pure JSON.

    # re.sub(pattern, replacement, string, flags=...)
    # r"^```(?:json)?\s*" matches:
    #   ^          → start of a line (with MULTILINE flag)
    #   ```        → three backticks
    #   (?:json)?  → optionally followed by "json"
    #   \s*        → any whitespace (including newline)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)

    # Remove the closing ``` fence:
    # \s*```$ matches optional whitespace then ``` at end of a line
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)

    text = text.strip()   # Clean up any remaining whitespace

    # ATTEMPT 2: Try parsing the cleaned text directly as JSON
    try:
        return json.loads(text)
        # json.loads() converts a JSON string to a Python dict.
        # If successful, we're done — return immediately.

    except json.JSONDecodeError:
        # JSON parsing failed. Try the last-resort extraction method below.
        pass

    # ATTEMPT 3 (last resort): Find the first {...} block in the text.
    # This handles cases where Gemini added text before/after the JSON object.
    # re.search(r"\{.*\}", text, re.DOTALL) finds the first {...} block.
    # re.DOTALL makes "." match newlines too (needed for multi-line JSON).
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
            # match.group() returns the full matched string (the {...} block)
        except json.JSONDecodeError:
            pass   # Even this attempt failed

    # All attempts failed — raise an informative error
    raise ValueError(
        f"Could not parse Gemini response as JSON.\n"
        f"First 300 chars:\n{text[:300]}"
        # Showing the first 300 characters helps debug what Gemini returned.
    )
