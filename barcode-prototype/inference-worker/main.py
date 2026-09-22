import base64
import io
import json
import os
import time

import cv2
import numpy as np
import torch
import zxingcpp
from kafka import KafkaConsumer, KafkaProducer
from PIL import Image, ImageDraw
from transformers import RTDetrV2ForObjectDetection, RTDetrImageProcessor

# --- Конфигурация ---
# Адрес Kafka-брокера. В docker-compose задаётся через переменную окружения.
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")

# Имя модели RT-DETR v2 на Hugging Face.
# r50vd — компромисс между точностью и скоростью на CPU.
MODEL_NAME = os.environ.get("MODEL_NAME", "PekingU/rtdetr_v2_r50vd")

# Порог confidence для детекций RT-DETR.
# Всё, что ниже, отбрасывается ещё до zxingcpp.
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.1"))

# Количество потоков torch. 0 = авто (torch сам выберет по числу ядер).
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", "0"))

# Максимальная сторона аннотированного фото (для экономии размера сообщения в Kafka).
ANNOT_MAX_SIDE = int(os.environ.get("ANNOT_MAX_SIDE", "1280"))

# Качество JPEG для аннотированного фото.
ANNOT_QUALITY = int(os.environ.get("ANNOT_QUALITY", "75"))


def log(*args):
    """Единая точка логирования с flush=True.

    Без flush строки могут висеть в буфере stdout и не появляться
    в `docker compose logs` до заполнения буфера.
    """
    print("[inference-worker]", *args, flush=True)


def setup_cpu():
    """Настраивает torch для работы на CPU.

    - ограничивает количество потоков (если задано TORCH_THREADS);
    - отключает autograd — для инференса градиенты не нужны.
    """
    if TORCH_THREADS > 0:
        torch.set_num_threads(TORCH_THREADS)
    torch.set_grad_enabled(False)
    log(f"torch threads: {torch.get_num_threads()} (cores: {os.cpu_count()})")


def wait_for_kafka():
    """Ждёт, пока Kafka станет доступна.

    Kafka в docker-compose стартует дольше, чем воркер, поэтому
    пробуем подключиться в цикле.
    """
    while True:
        try:
            c = KafkaConsumer(bootstrap_servers=KAFKA_BOOTSTRAP)
            c.close()
            log("kafka is up")
            return
        except Exception:
            log("waiting for kafka...")
            time.sleep(3)


def load_model(model_name: str):
    """Загружает RT-DETR v2 и процессор, прогревает модель.

    Прогрев (warmup) нужен, чтобы первый реальный кадр не тормозил
    из-за ленивой инициализации внутренних структур torch.
    """
    log("=== loading RT-DETR v2 (CPU) ===")
    log(f"model: {model_name}")

    t0 = time.time()
    processor = RTDetrImageProcessor.from_pretrained(model_name)
    log(f"processor loaded in {time.time() - t0:.1f}s")

    t0 = time.time()
    log("loading weights...")
    model = RTDetrV2ForObjectDetection.from_pretrained(model_name)
    log(f"weights loaded in {time.time() - t0:.1f}s")

    model.eval()

    t0 = time.time()
    dummy = Image.new("RGB", (640, 640), (0, 0, 0))
    inputs = processor(images=dummy, return_tensors="pt")
    with torch.no_grad():
        _ = model(**inputs)
    log(f"warmup done in {time.time() - t0:.1f}s")
    log("=== ready ===")
    return processor, model


def detect_objects(image: Image.Image, processor, model, threshold: float):
    """Прогоняет изображение через RT-DETR v2 и возвращает список детекций.

    Каждая детекция — словарь:
      label — индекс класса COCO (0..79);
      score — уверенность модели;
      box   — координаты [x0, y0, x1, y1] в пикселях исходного изображения.
    """
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=threshold
    )[0]

    detections = []
    for score, label, box in zip(
        results["scores"].tolist(),
        results["labels"].tolist(),
        results["boxes"].tolist(),
    ):
        detections.append(
            {
                "label": int(label),
                "score": float(score),
                "box": [float(x) for x in box],
            }
        )
    return detections


def decode_barcodes_crops(image: Image.Image, detections: list):
    """Прогоняет zxingcpp по кропам, вырезанным из детекций RT-DETR.

    Для каждого бокса:
      - вырезает область из изображения;
      - отправляет в zxingcpp.read_barcodes;
      - если штрихкод распознан — добавляет его текст в результат;
      - если нет — пишет «не удалось распознать».

    Возвращает список строк (по одному элементу на каждый бокс,
    плюс дополнительные строки, если в одном кропе нашлось несколько штрихкодов).
    """
    arr = np.array(image)
    h, w = arr.shape[:2]
    texts = []
    for d in detections:
        x0, y0, x1, y1 = [int(v) for v in d["box"]]
        # Обрезаем координаты по границам изображения,
        # чтобы не выйти за пределы массива.
        x0 = max(0, min(x0, w - 1))
        y0 = max(0, min(y0, h - 1))
        x1 = max(0, min(x1, w))
        y1 = max(0, min(y1, h))
        if x1 <= x0 or y1 <= y0:
            texts.append("не удалось распознать")
            continue
        crop = arr[y0:y1, x0:x1]
        if crop.size == 0:
            texts.append("не удалось распознать")
            continue
        try:
            found = zxingcpp.read_barcodes(crop)
            decoded = [b.text for b in found if b.text]
        except Exception as e:
            log(f"zxing error: {e}")
            decoded = []
        if decoded:
            texts.extend(decoded)
        else:
            texts.append("не удалось распознать")
    return texts


def draw_boxes(image: Image.Image, detections: list) -> str:
    """Рисует рамки детекций на копии изображения и возвращает base64 JPEG.

    Изображение предварительно уменьшается до ANNOT_MAX_SIDE по большей стороне —
    чтобы base64 не раздувал сообщение в Kafka.
    Координаты рамок масштабируются соответственно.
    """
    img = image.copy()
    orig_w, orig_h = img.size
    if max(orig_w, orig_h) > ANNOT_MAX_SIDE:
        scale = ANNOT_MAX_SIDE / max(orig_w, orig_h)
        img = img.resize((int(orig_w * scale), int(orig_h * scale)), Image.LANCZOS)
    else:
        scale = 1.0

    draw = ImageDraw.Draw(img)
    for d in detections:
        x0, y0, x1, y1 = [v * scale for v in d["box"]]
        draw.rectangle([x0, y0, x1, y1], outline="red", width=3)
        draw.text(
            (x0, max(0, y0 - 12)),
            f'{d["label"]} {d["score"]:.2f}',
            fill="red",
        )

    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=ANNOT_QUALITY, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def decode_image(b64_str: str) -> Image.Image:
    """Декодирует base64 (JPEG/PNG/...) в PIL.Image в RGB."""
    raw = base64.b64decode(b64_str)
    arr = np.frombuffer(raw, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def main():
    """Основной цикл воркера: читает кадры из box.frames, обрабатывает, пишет в frame.results."""
    setup_cpu()
    wait_for_kafka()
    processor, model = load_model(MODEL_NAME)

    # Consumer читает кадры, отправленные web-сервисом.
    # auto_offset_reset="earliest" — не терять сообщения, отправленные до старта воркера.
    consumer = KafkaConsumer(
        "box.frames",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="inference-worker",
        max_partition_fetch_bytes=20 * 1024 * 1024,
        fetch_max_bytes=50 * 1024 * 1024,
    )

    # Producer публикует результаты инференса.
    # max_request_size увеличен — в сообщении едет base64 аннотированного фото.
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        max_request_size=20 * 1024 * 1024,
    )

    log("started, waiting for frames...")
    for msg in consumer:
        data = msg.value
        box_id = data["box_id"]
        frame_index = data["frame_index"]
        total_frames = data["total_frames"]

        # Полный цикл обработки одного кадра:
        # 1. base64 -> PIL.Image
        # 2. RT-DETR -> список боксов
        # 3. zxingcpp по кропам боксов -> строки штрихкодов
        # 4. отрисовка рамок -> base64 JPEG
        t0 = time.time()
        img = decode_image(data["image_b64"])
        detections = detect_objects(img, processor, model, CONF_THRESHOLD)
        barcodes = decode_barcodes_crops(img, detections)
        annotated_b64 = draw_boxes(img, detections)
        dt = (time.time() - t0) * 1000

        log(
            f"box={box_id} frame={frame_index}/{total_frames} "
            f"-> {len(detections)} boxes, barcodes={barcodes} in {dt:.0f}ms"
        )

        # Отправляем результат дальше — в aggregator.
        producer.send(
            "frame.results",
            {
                "box_id": box_id,
                "frame_index": frame_index,
                "total_frames": total_frames,
                "detections": detections,
                "barcodes": barcodes,
                "annotated_image_b64": annotated_b64,
            },
        )
        producer.flush()

    producer.flush()


if __name__ == "__main__":
    main()