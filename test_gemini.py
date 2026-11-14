"""Send one short prompt to two Gemini models and print each reply or the full error."""
import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors

load_dotenv()  # reads GEMINI_API_KEY from .env
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

PROMPT = "Reply with the word OK"
MODELS = ["gemini-3.5-flash-lite", "gemini-3.8-flash"]


def list_models():
    print("  Models this key can use for generateContent:")
    for m in client.models.list():
        if "generateContent" in (m.supported_actions or []):
            print("   ", m.name.removeprefix("models/"))


listed = False
for model in MODELS:
    print(f"\n=== {model} ===")
    try:
        response = client.models.generate_content(model=model, contents=PROMPT)
        print("  Reply:", repr(response.text))
    except errors.APIError as e:
        print(f"  Error {e.code} {e.status}: {e.message}")
        if e.details:
            print("  Details:", e.details)
        if e.code == 404 and not listed:  # model name doesn't exist
            list_models()
            listed = True
    except Exception as e:
        print(f"  Error ({type(e).__name__}): {e}")
