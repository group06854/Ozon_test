import json
import os
import threading
import time

from kafka import KafkaConsumer, KafkaProducer

KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TIMEOUT_SECONDS = float(os.environ.get("AGGREGATOR_TIMEOUT", "15"))

state: dict[str, dict] = {}
state_lock = threading.Lock()


def log(*args):
    print("[aggregator]", *args, flush=True)


def wait_for_kafka():
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
    with state_lock:
        s = state.get(box_id)
        if not s or s["finalized"]:
            return
        s["finalized"] = True
        barcodes = sorted(s["barcodes"])
        detections = list(s["detections"])
        images = [s["images"][i] for i in sorted(s["images"])]

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
    wait_for_kafka()

    consumer = KafkaConsumer(
        "frame.results",
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        group_id="aggregator",
        max_partition_fetch_bytes=20 * 1024 * 1024,
        fetch_max_bytes=50 * 1024 * 1024,
    )
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        max_request_size=20 * 1024 * 1024,
    )

    threading.Thread(target=timeout_watcher, args=(producer,), daemon=True).start()

    log("started, waiting for frame results...")
    for msg in consumer:
        data = msg.value
        box_id = data["box_id"]
        frame_index = data["frame_index"]
        total_frames = data["total_frames"]

        barcodes_this_frame = list(data.get("barcodes") or [])
        detections_this_frame = list(data.get("detections") or [])

        log(
            f"got frame box={box_id} frame={frame_index}/{total_frames} "
            f"barcodes={barcodes_this_frame} "
            f"detections={len(detections_this_frame)} "
            f"has_image={'annotated_image_b64' in data}"
        )

        with state_lock:
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