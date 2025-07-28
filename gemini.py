import requests

GEMINI_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

def query_gemini_rest(signal_type, label, confidence, api_key):

    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": api_key,
    }

    prompt = (
        f"Explain the meaning of a {signal_type} signal classified as '{label}' "
        f"with a confidence of {confidence:.1%} in a medical diagnostic context."
    )

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        response = requests.post(GEMINI_ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()
        content = response.json()
        return content["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"⚠️ Gemini API error: {str(e)}"
