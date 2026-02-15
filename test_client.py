import requests
import pandas as pd
import json
import numpy as np
import os

def test_health():
    """Проверка health endpoint"""
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        print(f"🏥 Health check: {r.json()}")
        return r.status_code == 200
    except:
        print("❌ Сервер не запущен!")
        return False

def create_test_csv():
    """Создает тестовый CSV файл"""
    test_data = pd.DataFrame({
        'id': [1001, 1002, 1003, 1004, 1005],
        'age': [45, 62, 34, 55, 41],
        'gender': [1, 0, 1, 1, 0],
        'cholesterol': [180, 240, 150, 210, 190],
        'blood_pressure': [120, 140, 110, 135, 125],
        'heart_rate': [70, 85, 65, 80, 75],
        'smoking': [0, 1, 0, 1, 0],
        'diabetes': [0, 1, 0, 0, 1],
        'family_history': [1, 1, 0, 1, 0]
    })

    filename = 'test_heart_data.csv'
    test_data.to_csv(filename, index=False)
    print(f"✅ Создан тестовый файл: {filename}")
    return filename

def test_predict_csv():
    """Тестирование предсказания на CSV файле"""
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ CSV ПРЕДСКАЗАНИЙ")
    print("="*60)

    # Создаем тестовый CSV
    filename = create_test_csv()

    # Отправляем запрос
    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'text/csv')}
            r = requests.post("http://localhost:8000/predict_csv", files=files, timeout=30)

        if r.status_code == 200:
            # Проверяем тип ответа
            content_type = r.headers.get('content-type', '')

            if 'application/json' in content_type:
                result = r.json()
                print(f"\n✅ Успешно! Статус: {result['status']}")
                print(f"📊 Всего записей: {result['count']}")
                print(f"📈 Распределение:")
                print(f"   Класс 0: {result['distribution']['class_0']} ({result['distribution']['class_0_percent']:.1f}%)")
                print(f"   Класс 1: {result['distribution']['class_1']} ({result['distribution']['class_1_percent']:.1f}%)")
                print(f"\n🔮 Первые 5 предсказаний:")
                for i in range(min(5, len(result['predictions']))):
                    print(f"   ID {result['ids'][i]}: {result['predictions'][i]} (вероятность: {result['probabilities'][i]:.3f})")
            else:
                print(f"✅ HTML ответ получен (длина: {len(r.text)} символов)")
                print("💡 Это веб-страница, откройте её в браузере")

            return True
        else:
            print(f"❌ Ошибка: статус {r.status_code}")
            print(r.text)
            return False

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    finally:
        # Удаляем тестовый файл
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🧹 Удален тестовый файл: {filename}")

def test_predict_json():
    """Тестирование предсказания на JSON файле"""
    print("\n" + "="*60)
    print("🧪 ТЕСТИРОВАНИЕ JSON ПРЕДСКАЗАНИЙ")
    print("="*60)

    test_data = [
        {"id": 2001, "age": 45, "gender": 1, "cholesterol": 180, "blood_pressure": 120, "heart_rate": 70, "smoking": 0, "diabetes": 0, "family_history": 1},
        {"id": 2002, "age": 62, "gender": 0, "cholesterol": 240, "blood_pressure": 140, "heart_rate": 85, "smoking": 1, "diabetes": 1, "family_history": 1},
        {"id": 2003, "age": 34, "gender": 1, "cholesterol": 150, "blood_pressure": 110, "heart_rate": 65, "smoking": 0, "diabetes": 0, "family_history": 0}
    ]

    filename = 'test_heart_data.json'
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2)
    print(f"✅ Создан тестовый файл: {filename}")

    try:
        with open(filename, 'rb') as f:
            files = {'file': (filename, f, 'application/json')}
            r = requests.post("http://localhost:8000/predict_json", files=files, timeout=30)

        if r.status_code == 200:
            result = r.json()
            print(f"\n✅ Успешно! Статус: {result['status']}")
            print(f"📊 Всего записей: {result['count']}")
            print(f"\n🔮 Предсказания:")
            for i in range(len(result['predictions'])):
                print(f"   ID {result['ids'][i]}: {result['predictions'][i]} (вероятность: {result['probabilities'][i]:.3f})")
            return True
        else:
            print(f"❌ Ошибка: статус {r.status_code}")
            print(r.text)
            return False

    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            print(f"🧹 Удален тестовый файл: {filename}")

if __name__ == "__main__":
    print("🚀 КЛИЕНТ ДЛЯ ТЕСТИРОВАНИЯ API")
    print("="*60)

    if not test_health():
        print("\n💡 Запустите сервер командой:")
        print("   python app.py --port 8000")
        exit(1)

    test_predict_csv()
    test_predict_json()

    print("\n" + "="*60)
    print("✅ ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    print("\n🌐 Откройте в браузере: http://localhost:8000")
