import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Загружаем "базу знаний" — просто текстовый файл целиком
with open("knowledge.txt", "r", encoding="utf-8") as f:
    knowledge = f.read()

def ask_without_rag(question):
    """Задаём вопрос без контекста — модель отвечает 'из головы'"""
    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=question
    )
    return response.text

def ask_with_rag(question):
    """Задаём вопрос с контекстом из базы знаний"""
    prompt = f"""Ответь на вопрос, используя ТОЛЬКО информацию из контекста ниже.
Если в контексте нет ответа — честно скажи, что не знаешь.

Контекст:
{knowledge}

Вопрос: {question}"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text

print("RAG-сравнение запущено! Задавайте вопросы (или 'выход' чтобы закончить)\n")

while True:
    question = input("Вопрос: ")
    if question.lower() in ["выход", "exit", "quit"]:
        print("Пока!")
        break

    if not question.strip():
        print("Кажется, вы ничего не написали.\n")
        continue

    print("\n=== БЕЗ RAG ===")
    print(ask_without_rag(question))

    print("\n=== С RAG ===")
    print(ask_with_rag(question))
    print("\n" + "-"*50 + "\n")