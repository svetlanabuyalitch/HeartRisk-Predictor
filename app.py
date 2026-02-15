from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import argparse
import logging
import pandas as pd
import numpy as np
import joblib
import os
import json
from model import Model

app = FastAPI()

# Создаем папки если их нет
os.makedirs("tmp", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/tmp", StaticFiles(directory="tmp"), name='images')
templates = Jinja2Templates(directory="templates")

# Настройка логирования
app_logger = logging.getLogger(__name__)
app_logger.setLevel(logging.INFO)
app_handler = logging.StreamHandler()
app_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")
app_handler.setFormatter(app_formatter)
app_logger.addHandler(app_handler)

# Глобальная переменная для модели
model = None

@app.on_event("startup")
async def load_model():
    """Загружает модель при запуске сервера"""
    global model
    model_path = "best_heart_attack_model.pkl"
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            app_logger.info(f"✅ Модель загружена из {model_path}")
        else:
            app_logger.warning(f"⚠️ Модель не найдена по пути {model_path}")
            app_logger.warning("Используется заглушка для демонстрации")
            model = None
    except Exception as e:
        app_logger.error(f"❌ Ошибка загрузки модели: {e}")
        model = None

@app.get("/health")
def health():
    """Проверка работоспособности сервера"""
    return {"status": "OK", "model_loaded": model is not None}

@app.get("/")
def main(request: Request):
    """Главная страница с формой загрузки"""
    return templates.TemplateResponse("predict_form.html",
                                      {"request": request})

@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...), request: Request = None):
    """
    Принимает CSV файл с тестовой выборкой и возвращает предсказания
    """
    save_pth = f"tmp/{file.filename}"
    app_logger.info(f'📁 Обработка CSV файла - {save_pth}')

    # Сохраняем загруженный файл
    with open(save_pth, "wb") as fid:
        fid.write(await file.read())

    try:
        # Загружаем CSV
        df = pd.read_csv(save_pth)
        app_logger.info(f"📊 CSV загружен. Размер: {df.shape}")

        # Проверяем наличие id
        if 'id' in df.columns:
            ids = df['id'].values
            X = df.drop(['id'], axis=1)
        else:
            ids = np.arange(len(df))
            X = df

        # Делаем предсказания
        if model is not None:
            predictions = model.predict(X)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(X)[:, 1]
            else:
                probabilities = predictions.astype(float)
        else:
            # Заглушка для демонстрации
            np.random.seed(42)
            predictions = np.random.randint(0, 2, size=len(df))
            probabilities = np.random.rand(len(df))

        # Формируем результат
        result = {
            "status": "success",
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "ids": ids.tolist() if isinstance(ids, np.ndarray) else ids,
            "count": len(predictions),
            "distribution": {
                "class_0": int((predictions == 0).sum()),
                "class_1": int((predictions == 1).sum()),
                "class_0_percent": float((predictions == 0).mean() * 100),
                "class_1_percent": float((predictions == 1).mean() * 100)
            }
        }

        # Сохраняем результаты
        result_path = f"tmp/result_{file.filename.replace('.csv', '')}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        app_logger.info(f"✅ Предсказания выполнены. Всего: {len(predictions)} записей")

        # Возвращаем HTML страницу с результатами
        if request:
            return templates.TemplateResponse(
                "result.html",
                {
                    "request": request,
                    "status": "success",
                    "total": len(predictions),
                    "class_0": result["distribution"]["class_0"],
                    "class_1": result["distribution"]["class_1"],
                    "class_0_percent": f"{result['distribution']['class_0_percent']:.1f}",
                    "class_1_percent": f"{result['distribution']['class_1_percent']:.1f}",
                    "predictions": predictions[:10].tolist(),
                    "ids": ids[:10].tolist(),
                    "result_path": result_path,
                    "model_loaded": model is not None
                }
            )

        return result

    except Exception as e:
        app_logger.error(f"❌ Ошибка обработки CSV: {str(e)}")
        if request:
            return templates.TemplateResponse(
                "error.html",
                {"request": request, "error": str(e)}
            )
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Скачивание файла с результатами"""
    file_path = f"tmp/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type='application/json', filename=filename)
    raise HTTPException(status_code=404, detail="Файл не найден")

@app.get("/model_info")
def model_info():
    """Информация о загруженной модели"""
    if model is None:
        return {"status": "no_model", "message": "Модель не загружена"}

    info = {
        "status": "loaded",
        "model_type": type(model).__name__,
    }

    if hasattr(model, 'n_features_in_'):
        info["n_features"] = model.n_features_in_

    if hasattr(model, 'classes_'):
        info["classes"] = model.classes_.tolist()

    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default=8000, type=int, dest="port")
    parser.add_argument("--host", default="0.0.0.0", type=str, dest="host")
    args = vars(parser.parse_args())

    uvicorn.run(app, **args)
