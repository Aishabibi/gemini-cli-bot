import os
from dotenv import load_dotenv
from google import genai
import ollama
from datetime import datetime

def calculate(expression: str) -> str:
    """Вычисляет математическое выражение и возвращает результат.
    Например: '2 + 2', '15 * 3', '100 / 4'."""
    print(f"🔧 [Функция calculate() была вызвана с выражением: {expression}]")
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Ошибка вычисления: {e}"

def get_current_time() -> str:
    """Возвращает текущее время."""
    print("🔧 [Функция get_current_time() была вызвана моделью]")
    return datetime.now().strftime("%H:%M:%S, %d.%m.%Y")

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

PROVIDER = "ollama"

SYSTEM_PROMPT = "Ты дружелюбный ментор для начинающих AI-инженеров. Отвечай коротко, максимум 3 предложения, без длинных списков."

print(f"Бот запущен ({PROVIDER})! Напишите что-нибудь (или 'выход' чтобы закончить)\n")

history = []

while True:
    user_input = input("Вы: ")
    if user_input.lower() in ["выход", "exit", "quit"]:
        print("Пока!")
        break

    if not user_input.strip():
        print("Бот: Кажется, вы ничего не написали — попробуйте ещё раз.\n")
        continue

    if PROVIDER == "gemini":
        history.append({"role": "user", "parts": [{"text": user_input}]})
        history = history[-10:]

        try:
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=history,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "tools": [get_current_time, calculate]
                }
            )
            answer = response.text
        except Exception as e:
            print(f"Бот: Ой, сервер сейчас недоступен или перегружен. Попробуйте через минуту.\n(Техническая причина: {e})\n")
            history.pop()
            continue

        history.append({"role": "model", "parts": [{"text": answer}]})

    elif PROVIDER == "ollama":
        history.append({"role": "user", "content": user_input})
        history = history[-10:]

        try:
            response = ollama.chat(
                model="llama3.2",
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history,
                tools=[get_current_time, calculate]
            )

            message = response["message"]

            if message.get("tool_calls"):
                for tool_call in message["tool_calls"]:
                    func_name = tool_call["function"]["name"]
                    func_args = tool_call["function"]["arguments"]

                    if func_name == "get_current_time":
                        result = get_current_time()
                        print(f"➡️ Функция вернула: {result}")
                    elif func_name == "calculate":
                        result = calculate(func_args.get("expression", ""))
                    else:
                        result = "Неизвестная функция"

                    history.append({"role": "assistant", "content": "", "tool_calls": message["tool_calls"]})
                    history.append({"role": "tool", "content": result})

                    final_response = ollama.chat(
                        model="llama3.2",
                        messages=[{"role": "system", "content": SYSTEM_PROMPT}] + history
                    )
                    answer = final_response["message"]["content"]
            else:
                answer = message["content"]

            history.append({"role": "assistant", "content": answer})

        except Exception as e:
            print(f"Бот: Ошибка при обращении к Ollama: {e}\n")
            history.pop()
            continue

    print(f"Бот: {answer}\n")
