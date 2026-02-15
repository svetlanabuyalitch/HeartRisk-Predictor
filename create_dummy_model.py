import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def create_dummy_model():
    """Создает простую модель для демонстрации"""
    print("🤖 Создание демо-модели...")

    # Создаем искусственные данные
    np.random.seed(42)
    n_samples = 1000
    n_features = 8

    X = np.random.randn(n_samples, n_features)

    # Создаем целевую переменную с некоторой зависимостью
    y = (X[:, 0] + X[:, 1] * 0.5 + X[:, 2] * 0.3 + np.random.randn(n_samples) * 0.5 > 0).astype(int)

    # Создаем и обучаем модель
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    # Сохраняем модель
    model_path = 'best_heart_attack_model.pkl'
    joblib.dump(model, model_path)

    print(f"✅ Модель создана и сохранена в {model_path}")
    print(f"📊 Форма обучения: {X.shape}")
    print(f"🎯 Точность на обучающих данных: {model.score(X, y):.3f}")
    print(f"🔮 Ожидаемое количество признаков: {model.n_features_in_}")

    # Создаем тестовый CSV файл
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

    test_data.to_csv('test_sample.csv', index=False)
    print("📁 Создан тестовый файл: test_sample.csv")

    return model

if __name__ == "__main__":
    create_dummy_model()

    print("\n" + "="*50)
    print("🚀 ДЛЯ ЗАПУСКА СЕРВЕРА:")
    print("="*50)
    print("python app.py --port 8000")
    print("\n🌐 ДЛЯ ТЕСТИРОВАНИЯ:")
    print("python test.py")
    print("python test_client.py")
    print("\n📂 Откройте в браузере:")
    print("http://localhost:8000")
