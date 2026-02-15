import requests
import sys

def main():
    """Проверка работоспособности сервера"""
    url = "http://localhost:8000/health"

    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            print(f"❌ Ошибка: статус {r.status_code}")
            print(r.text)
            sys.exit(1)

        data = r.json()
        print(f"✅ Сервер работает!")
        print(f"📊 Статус: {data['status']}")
        print(f"🤖 Модель загружена: {data['model_loaded']}")

    except requests.exceptions.ConnectionError:
        print("❌ Ошибка подключения к серверу!")
        print("💡 Убедитесь что сервер запущен: python app.py --port 8000")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Неожиданная ошибка: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
