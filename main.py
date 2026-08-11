import os
from dotenv import load_dotenv
from google import genai
import ollama

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROVIDER = "gemini"

SYSTEM_PROMPT = "Ты дружелюбный ментор для начинающих AI-инженеров. Отвечай коротко, максимум 3 предложения, без длинных списков."

print(f"Бот запущен ({PROVIDER})! Напишите что-нибудь (или 'выход' чтобы закончить)\n")

history = []

while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["выход", "exit", "quit"]:
        print("Пока!")
        break

    if PROVIDER == "gemini":
        history.append({"role": "user", "parts": [{"text": user_input}]})
        history = history[-10:]

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=history,
            config={"system_instruction": SYSTEM_PROMPT}
        )
        answer = response.text
        history.append({"role": "model", "parts": [{"text": answer}]})

    elif PROVIDER == "ollama":
        history.append({"role": "user", "content": user_input})
        history = history[-10:]

        response = ollama.chat(
            model="llama3.2",
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history
        )
        answer = response["message"]["content"]
        history.append({"role": "assistant", "content": answer})

    print(f"Бот: {answer}\n")