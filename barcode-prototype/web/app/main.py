import base64
import json
import os
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from kafka import KafkaProducer, KafkaConsumer

# --- Конфигурация ---
# Адрес Kafka-брокера. В docker-compose задаётся через переменную окружения.
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")

# FastAPI-приложение. Через /static раздаётся фронт (index.html и прочее).
app = FastAPI(title="Barcode Prototype")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

# --- Состояние результатов ---
# Ключ — box_id, значение — словарь со статусом и данными по этому боксу:
#   status        — "pending" (ждём) или "done" (результат получен);
#   barcodes      — список строк со штрихкодами (от zxingcpp);
#   detections    — список боксов от RT-DETR;
#   images        — список base64-строк с аннотированными фото;
#   fallback_used — флаг, использовался ли резервный сценарий обработки.
results: dict[str, dict] = {}
results_lock = threading.Lock()

# Producer для отправки кадров в Kafka. Создаётся лениво — один раз на процесс.
_producer = None
_producer_lock = threading.Lock()


def log(*args):
    """Единая точка логирования с flush=True — чтобы строки сразу шли в docker logs."""
    print("[web]", *args, flush=True)


def get_producer() -> KafkaProducer:
    """Ленивая инициализация KafkaProducer.

    Producer создаётся один раз и переиспользуется всеми запросами
    (создание producer'а — дорогая операция, незачем делать это на каждый upload).
    """
    global _producer
    with _producer_lock:
        if _producer is None:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                # Увеличен, потому что в сообщении едет base64 фото.
                max_request_size=20 * 1024 * 1024,
            )
        return _producer


def consume_box_results():
    """Фоновый поток: читает финальные результаты из box.results и кэширует их в памяти.

    Работает вечно; при ошибке подключения — переподключается через 3 секунды.
    Результаты сохраняются в `results` по ключу box_id и отдаются в GET /boxes/{box_id}/result.
    """
    while True:
        try:
            consumer = KafkaConsumer(
                "box.results",
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
                # earliest — не терять сообщения, отправленные до старта web.
                auto_offset_reset="earliest",
                group_id="web-results-reader",
                max_partition_fetch_bytes=20 * 1024 * 1024,
                fetch_max_bytes=50 * 1024 * 1024,
            )
            log("connected to kafka, listening for box.results")
            for msg in consumer:
                data = msg.value
                box_id = data["box_id"]
                with results_lock:
                    results[box_id] = {
                        "status": "done",
                        "barcodes": data.get("barcodes", []),
                        "detections": data.get("detections", []),
                        "images": data.get("images", []),
                        "fallback_used": data.get("fallback_used", False),
                    }
                log(
                    f"stored box={box_id} "
                    f"barcodes={len(results[box_id]['barcodes'])} "
                    f"detections={len(results[box_id]['detections'])} "
                    f"images={len(results[box_id]['images'])}"
                )
        except Exception as e:
            log(f"kafka consumer error, retrying in 3s: {e}")
            time.sleep(3)


# Запускаем фоновый reader сразу при старте приложения.
threading.Thread(target=consume_box_results, daemon=True).start()


@app.get("/")
def index():
    """Отдаёт главную страницу фронта."""
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/boxes/upload")
async def upload_box(files: list[UploadFile] = File(...)):
    """Принимает одно или несколько фото, отправляет их в Kafka, возвращает box_id.

    Логика:
      1. Генерируем уникальный box_id.
      2. Создаём в results запись со статусом "pending".
      3. Каждое фото кодируем в base64 и отправляем в топик box.frames
         с указанием frame_index и total_frames.
      4. Возвращаем box_id — фронт по нему будет опрашивать результат.
    """
    box_id = str(uuid.uuid4())
    total = len(files)
    producer = get_producer()

    # Сразу регистрируем box_id со статусом "pending",
    # чтобы GET /result не вернул "unknown" до прихода результата.
    with results_lock:
        results[box_id] = {
            "status": "pending",
            "barcodes": [],
            "detections": [],
            "images": [],
            "fallback_used": False,
        }

    for idx, f in enumerate(files):
        content = await f.read()
        b64 = base64.b64encode(content).decode("ascii")
        producer.send(
            "box.frames",
            {
                "box_id": box_id,
                "frame_index": idx,
                "total_frames": total,
                "filename": f.filename,
                "image_b64": b64,
            },
        )
    producer.flush()
    log(f"uploaded box={box_id} frames={total}")
    return {"box_id": box_id, "frames_sent": total}


@app.get("/boxes/{box_id}/result")
def get_result(box_id: str):
    """Отдаёт текущий статус и результат обработки по box_id.

    Возможные ответы:
      {"status": "unknown"} — такого box_id нет (или сервер перезапускался);
      {"status": "pending", ...} — обработка ещё идёт;
      {"status": "done", "barcodes": [...], "detections": [...], "images": [...]} — готово.
    """
    with results_lock:
        r = results.get(box_id)
    if r is None:
        return {"status": "unknown"}
    return r