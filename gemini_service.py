from dotenv import load_dotenv
import os
from google import genai

load_dotenv()
client = genai.Client(api_key=os.environ.get('GEMINI_API_KEY'))
def answer(prompt: str, best_doc):
    system_instruction = (
        f"You are an AI assistent. Answer based ONLY on this document. {best_doc["text"]}"
    )

    full_prompt = f"{system_instruction}\n\n{prompt}"

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=full_prompt,
    )

    return response.text

