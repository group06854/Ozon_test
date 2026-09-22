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

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")

app = FastAPI(title="Barcode Prototype")

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")

results: dict[str, dict] = {}
results_lock = threading.Lock()

_producer = None
_producer_lock = threading.Lock()


def log(*args):
    print("[web]", *args, flush=True)


def get_producer() -> KafkaProducer:
    global _producer
    with _producer_lock:
        if _producer is None:
            _producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
                max_request_size=20 * 1024 * 1024,
            )
        return _producer


def consume_box_results():
    while True:
        try:
            consumer = KafkaConsumer(
                "box.results",
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_deserializer=lambda v: json.loads(v.decode("utf-8")),
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


threading.Thread(target=consume_box_results, daemon=True).start()


@app.get("/")
def index():
    return FileResponse(BASE_DIR / "static" / "index.html")


@app.post("/boxes/upload")
async def upload_box(files: list[UploadFile] = File(...)):
    box_id = str(uuid.uuid4())
    total = len(files)
    producer = get_producer()

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
    with results_lock:
        r = results.get(box_id)
    if r is None:
        return {"status": "unknown"}
    return r