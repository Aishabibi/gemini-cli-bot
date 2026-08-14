import os
from dotenv import load_dotenv
from google import genai
import numpy as np

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values)

def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Загружаем базу знаний и разбиваем на куски (по строкам)
with open("knowledge.txt", "r", encoding="utf-8") as f:
    chunks = [line.strip() for line in f if line.strip()]

print(f"Загружено {len(chunks)} кусков базы знаний. Считаем эмбеддинги...\n")

chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

def find_relevant_chunks(question, top_n=2):
    """Находит top_n самых похожих кусков на вопрос"""
    question_embedding = get_embedding(question)
    similarities = [cosine_similarity(question_embedding, emb) for emb in chunk_embeddings]

    # Сортируем куски по похожести, берём top_n
    ranked = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

def ask_with_smart_rag(question):
    relevant = find_relevant_chunks(question)

    print("🔍 Найденные релевантные куски:")
    for chunk, score in relevant:
        print(f"  ({score:.4f}) {chunk}")

    context = "\n".join([chunk for chunk, score in relevant])

    prompt = f"""Ответь на вопрос, используя ТОЛЬКО информацию из контекста ниже.
Если в контексте нет ответа — честно скажи, что не знаешь.

Контекст:
{context}

Вопрос: {question}"""

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt
    )
    return response.text

print("Умный RAG с поиском по кускам запущен! Задавайте вопросы (или 'выход')\n")

while True:
    question = input("Вопрос: ")
    if question.lower() in ["выход", "exit", "quit"]:
        print("Пока!")
        break
    if not question.strip():
        continue

    print()
    answer = ask_with_smart_rag(question)
    print(f"\nОтвет: {answer}\n")
    print("-" * 50 + "\n")