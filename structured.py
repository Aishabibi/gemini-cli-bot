import os
import json
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Схема — описываем, какую структуру данных хотим получить
schema = {
    "type": "object",
    "properties": {
        "кто": {"type": "string"},
        "дата": {"type": "string"},
        "время": {"type": "string"},
        "тема": {"type": "string"}
    },
    "required": ["кто", "дата", "время", "тема"]
}

text = "Может, увидимся на следующей неделе, обсудим что-нибудь"

response = client.models.generate_content(
    model="gemini-flash-latest",
    contents=f"Извлеки структурированные данные из текста: {text}",
    config={
        "response_mime_type": "application/json",
        "response_schema": schema
    }
)

print("Сырой ответ модели:")
print(response.text)

print("\nПреобразовано в Python-словарь:")
data = json.loads(response.text)
print(data)
print(f"\nТип данных: {type(data)}")
print(f"Тема встречи: {data['тема']}")


# Проверка на полноту данных (простая версия Human-in-the-Loop)
missing_fields = [key for key, value in data.items() if value == "не указано"]

if missing_fields:
    print(f"\n⚠️ Внимание: не удалось извлечь поля: {', '.join(missing_fields)}")
    print("Требуется проверка человеком перед сохранением в базу данных.")
else:
    print("\n✅ Все данные извлечены полностью, можно сохранять автоматически.")