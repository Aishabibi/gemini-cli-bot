import os
import time
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

with open("course_full.txt", "r", encoding="utf-8") as f:
    full_text = f.read()

chunks = [c.strip() for c in full_text.split("\n\n") if c.strip()]

print(f"Файл разбит на {len(chunks)} кусков.\n")

def naive_rag(question):
    start = time.time()
    prompt = f"Ответь на вопрос, используя контекст:\n\n{full_text}\n\nВопрос: {question}"
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    elapsed = time.time() - start
    return response.text, elapsed, len(prompt)

print("Считаем эмбеддинги для всех кусков (может занять время)...")
chunk_embeddings = []
for i, c in enumerate(chunks):
    chunk_embeddings.append(get_embedding(c))
    print(f"  Обработано {i+1}/{len(chunks)}")
    time.sleep(0.7)
print("Готово!\n")

def smart_rag(question, top_n=3):
    start = time.time()
    q_emb = get_embedding(question)
    similarities = [cosine_similarity(q_emb, emb) for emb in chunk_embeddings]
    ranked = sorted(zip(chunks, similarities), key=lambda x: x[1], reverse=True)
    top_chunks = [c for c, s in ranked[:top_n]]
    context = "\n\n".join(top_chunks)

    prompt = f"Ответь на вопрос, используя контекст:\n\n{context}\n\nВопрос: {question}"
    response = client.models.generate_content(model="gemini-flash-latest", contents=prompt)
    elapsed = time.time() - start
    return response.text, elapsed, len(prompt)

question = input("Ваш вопрос про курс: ")

print("\n=== НАИВНЫЙ RAG (весь файл целиком) ===")
answer1, time1, size1 = naive_rag(question)
print(answer1)
print(f"\n⏱ Время: {time1:.2f}с | 📏 Размер промпта: {size1} символов")

print("\n" + "="*50)

print("\n=== УМНЫЙ RAG (только релевантные куски) ===")
answer2, time2, size2 = smart_rag(question)
print(answer2)
print(f"\n⏱ Время: {time2:.2f}с | 📏 Размер промпта: {size2} символов")