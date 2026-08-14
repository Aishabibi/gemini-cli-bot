import os
from dotenv import load_dotenv
from google import genai
import numpy as np

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def get_embedding(text):
    """Превращает текст в вектор (список чисел)"""
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return np.array(result.embeddings[0].values)

def cosine_similarity(vec1, vec2):
    """Считает, насколько два вектора похожи (от -1 до 1, чем ближе к 1 - тем похожее)"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

# Тестовые фразы
phrases = [
    "Собака бежит по парку",
    "Пёс гуляет на улице",
    "Я изучаю программирование на Python",
    "Ракета летит в космос"
]

print("Считаем эмбеддинги для каждой фразы...\n")
embeddings = [get_embedding(phrase) for phrase in phrases]

# Сравниваем первую фразу со всеми остальными
base_phrase = phrases[0]
base_embedding = embeddings[0]

print(f"Сравниваем с фразой: '{base_phrase}'\n")

for i in range(1, len(phrases)):
    similarity = cosine_similarity(base_embedding, embeddings[i])
    print(f"'{phrases[i]}' → похожесть: {similarity:.4f}")