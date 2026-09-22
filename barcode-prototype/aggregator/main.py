import json
import os
import threading
import time

from kafka import KafkaConsumer, KafkaProducer

# --- Конфигурация ---
# Адрес Kafka-брокера. В docker-compose передаётся через переменную окружения.
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")

# Сколько секунд ждать недостающие кадры одного box_id, прежде чем закрыть его
# по таймауту (на случай потери/задержки отдельных кадров).
TIMEOUT_SECONDS = float(os.environ.get("AGGREGATOR_TIMEOUT", "15"))

# --- Состояние агрегатора ---
# Ключ — box_id, значение — словарь с накопленными данными по этому боксу:
#   total        — сколько кадров ожидается всего (total_frames);
#   received     — множество индексов уже полученных кадров;
#   barcodes     — множество распознанных строк (zxingcpp) по всем кадрам;
#   detections   — список всех детекций (боксов) от RT-DETR;
#   images       — словарь frame_index -> base64 аннотированного фото;
#   last_update  — время последнего обновления (для контроля таймаута);
#   finalized    — флаг, что результат уже отправлен в box.results.
state: dict[str, dict] = {}
state_lock = threading.Lock()


def log(*args):
    """Единая точка логирования с принудительным flush.

    flush=True нужен, чтобы строки сразу попадали в `docker compose logs`,
    а не висели в буфере stdout.
    """
    print("[aggregator]", *args, flush=True)


def wait_for_kafka():
    """Ждём, пока Kafka станет доступна.

    Kafka в docker-compose может стартовать дольше, чем aggregator.
    Пробуем подключиться в цикле, пока не получится.
    """
    while True:
        try:
            c = KafkaConsumer(bootstrap_servers=KAFKA_BOOTSTRAP)
            c.close()
            log("kafka is up")
            return
        except Exception as e:
            log(f"waiting for kafka... ({e})")
            time.sleep(3)


def finalize(box_id: str, producer: KafkaProducer):
    """Финализирует бокс и публикует итог в топик box.results.

    Вызывается либо когда собраны все кадры (received >= total),
    либо по таймауту из timeout_watcher.
    Гарантирует, что для одного box_id результат будет отправлен ровно один раз.
    """
    with state_lock:
        s = state.get(box_id)
        if not s or s["finalized"]:
            return
        s["finalized"] = True
        barcodes = sorted(s["barcodes"])
        detections = list(s["detections"])
        # images — словарь frame_index -> b64; сортируем по индексу,
        # чтобы порядок фото соответствовал порядку кадров.
        images = [s["images"][i] for i in sorted(s["images"])]

    # Итоговое сообщение для web — всё, что нужно фронту.
    payload = {
        "box_id": box_id,
        "barcodes": barcodes,
        "detections": detections,
        "images": images,
        "fallback_used": False,
    }
    producer.send("box.results", payload)
    producer.flush()
    log(
        f"box={box_id} FINALIZED -> "
        f"barcodes={barcodes} detections={len(detections)} images={len(images)}"
    )


def timeout_watcher(producer: KafkaProducer):
    """Фоновый поток: закрывает по таймауту боксы, которые так и не собрали все кадры.

    Нужен на случай, если часть кадров потерялась (например, inference-worker
    не смог обработать фото), чтобы web не ждал результат вечно.
    """
    while True:
        time.sleep(2)
        now = time.time()
        with state_lock:
            due = [
                box_id
                for box_id, s in state.items()
                if not s["finalized"] and now - s["last_update"] > TIMEOUT_SECONDS
            ]
        for box_id in due:
            with state_lock:
                s = state.get(box_id)
                if s:
                    log(
                        f"box={box_id} TIMEOUT "
                        f"(received {len(s['received'])}/{s['total']})"
                    )
            finalize(box_id, producer)


def main():
    """Основной цикл: читает frame.results, копит данные по box_id, финализирует."""
    wait_for_kafka()

    # Consumer читает результаты инференса от inference-worker.
    # auto_offset_reset="earliest" — чтобы не терять сообщения,
    # отправленные до старта агрегатора.
    consumer = KafkaConsumer(
        "frame.results",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="aggregator",
        max_partition_fetch_bytes=20 * 1024 * 1024,
        fetch_max_bytes=50 * 1024 * 1024,
    )

    # Producer публикует финальные результаты в box.results.
    # max_request_size увеличен, потому что в сообщении едет base64 аннотированного фото.
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        max_request_size=20 * 1024 * 1024,
    )

    # Отдельный поток следит за таймаутами незакрытых боксов.
    threading.Thread(target=timeout_watcher, args=(producer,), daemon=True).start()

    log("started, waiting for frame results...")
    for msg in consumer:
        data = msg.value
        box_id = data["box_id"]
        frame_index = data["frame_index"]
        total_frames = data["total_frames"]

        # barcodes — строки от zxingcpp (или "не удалось распознать").
        # detections — боксы от RT-DETR.
        # Берём как есть, ничего не пересобираем и не подменяем.
        barcodes_this_frame = list(data.get("barcodes") or [])
        detections_this_frame = list(data.get("detections") or [])

        log(
            f"got frame box={box_id} frame={frame_index}/{total_frames} "
            f"barcodes={barcodes_this_frame} "
            f"detections={len(detections_this_frame)} "
            f"has_image={'annotated_image_b64' in data}"
        )

        with state_lock:
            # Первый кадр по box_id создаёт запись в state.
            s = state.setdefault(
                box_id,
                {
                    "total": total_frames,
                    "received": set(),
                    "barcodes": set(),
                    "detections": [],
                    "images": {},
                    "last_update": time.time(),
                    "finalized": False,
                },
            )
            s["received"].add(frame_index)
            s["barcodes"].update(barcodes_this_frame)
            s["detections"].extend(detections_this_frame)
            if "annotated_image_b64" in data:
                s["images"][frame_index] = data["annotated_image_b64"]
            s["last_update"] = time.time()
            # Все кадры получены — можно финализировать, не дожидаясь таймаута.
            done = len(s["received"]) >= s["total"]
            log(
                f"box={box_id} state: received={len(s['received'])}/{s['total']} "
                f"barcodes={len(s['barcodes'])} detections={len(s['detections'])} "
                f"images={len(s['images'])} done={done}"
            )

        if done:
            finalize(box_id, producer)


if __name__ == "__main__":
    main()