import cv2
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import easyocr
from ultralytics import YOLO


# =========================================================
# CONFIG
# =========================================================
VIDEO_SOURCE = "helmet.mp4"
OUTPUT_DIR = Path("outputs")
CROPS_DIR = OUTPUT_DIR / "plate_crops"
FRAMES_DIR = OUTPUT_DIR / "evidence_frames"
LOG_JSON = OUTPUT_DIR / "violations.json"

OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
CROPS_DIR.mkdir(exist_ok=True, parents=True)
FRAMES_DIR.mkdir(exist_ok=True, parents=True)

CUSTOM_MODEL_PATH = "person_helmet_vehicle_plate.pt"
FALLBACK_MODEL_PATH = "yolov8n.pt"

TRACKER_CFG = "bytetrack.yaml"

ENABLE_EMAIL = False
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_EMAIL = "your_email@gmail.com"
SMTP_PASSWORD = "your_app_password"
ALERT_TO = "reviewer@example.com"

HELMET_MISSING_FRAMES = 10

PERSON_CLASS = "person"
HELMET_CLASS = "helmet"
PLATE_CLASS = "number_plate"
VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}

OCR_LANGS = ["en"]


# =========================================================
# HELPERS
# =========================================================
def send_review_email(subject: str, body: str) -> None:
    if not ENABLE_EMAIL:
        print("\n[EMAIL DISABLED]")
        print(subject)
        print(body)
        return

    msg = MIMEMultipart()
    msg["From"] = SMTP_EMAIL
    msg["To"] = ALERT_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
    server.starttls()
    server.login(SMTP_EMAIL, SMTP_PASSWORD)
    server.sendmail(SMTP_EMAIL, ALERT_TO, msg.as_string())
    server.quit()


def iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def box_from_xyxy(xyxy) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = map(int, xyxy)
    return x1, y1, x2, y2


def clamp_crop(frame: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, w - 1))
    x2 = max(1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(1, min(y2, h))
    return frame[y1:y2, x1:x2].copy()


def plate_ocr(reader: easyocr.Reader, plate_crop: np.ndarray) -> str:
    results = reader.readtext(plate_crop, detail=0)
    if not results:
        return ""
    return "".join(results).replace(" ", "").upper()


def save_json(data: List[Dict], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_model_with_fallback() -> tuple[YOLO, bool]:
    custom_path = Path(CUSTOM_MODEL_PATH)

    if custom_path.exists():
        print(f"Using custom model: {custom_path}")
        return YOLO(str(custom_path)), True

    print(f"Custom model not found: {custom_path}")
    print(f"Falling back to pretrained model: {FALLBACK_MODEL_PATH}")
    print("Helmet and number plate detection will be disabled unless you train a custom model.")
    return YOLO(FALLBACK_MODEL_PATH), False


# =========================================================
# MAIN PIPELINE
# =========================================================
def run_pipeline():
    model, using_custom_model = load_model_with_fallback()
    reader = easyocr.Reader(OCR_LANGS, gpu=False)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {VIDEO_SOURCE}")

    violation_memory: Dict[int, int] = {}
    reported_tracks = set()
    violation_log: List[Dict] = []

    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    names = model.names
    available_classes = set(names.values()) if isinstance(names, dict) else set(names)

    helmet_supported = HELMET_CLASS in available_classes
    plate_supported = PLATE_CLASS in available_classes

    print("Available model classes sample:", list(available_classes)[:20])
    print("Helmet supported:", helmet_supported)
    print("Plate supported:", plate_supported)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        timestamp_sec = frame_idx / fps

        results = model.track(
            source=frame,
            persist=True,
            tracker=TRACKER_CFG,
            conf=0.25,
            verbose=False
        )

        if not results or results[0].boxes is None:
            continue

        r = results[0]
        boxes = r.boxes

        persons = []
        helmets = []
        vehicles = []
        plates = []

        for b in boxes:
            cls_id = int(b.cls[0].item())
            conf = float(b.conf[0].item())
            xyxy = box_from_xyxy(b.xyxy[0].tolist())
            track_id = int(b.id[0].item()) if b.id is not None else -1

            cls_name = names[cls_id] if isinstance(names, dict) else names[cls_id]

            item = {
                "class_id": cls_id,
                "class_name": cls_name,
                "conf": conf,
                "box": xyxy,
                "track_id": track_id,
            }

            if cls_name == PERSON_CLASS:
                persons.append(item)
            elif helmet_supported and cls_name == HELMET_CLASS:
                helmets.append(item)
            elif cls_name in VEHICLE_CLASSES:
                vehicles.append(item)
            elif plate_supported and cls_name == PLATE_CLASS:
                plates.append(item)

        # Helmet logic only if model supports helmet
        if helmet_supported:
            for person in persons:
                pbox = person["box"]
                pid = person["track_id"]

                has_helmet = False
                for helmet in helmets:
                    hbox = helmet["box"]
                    if iou(pbox, hbox) > 0.05:
                        has_helmet = True
                        break

                if not has_helmet and pid != -1:
                    violation_memory[pid] = violation_memory.get(pid, 0) + 1
                else:
                    if pid in violation_memory:
                        violation_memory[pid] = 0

                if (
                    pid != -1
                    and violation_memory.get(pid, 0) >= HELMET_MISSING_FRAMES
                    and pid not in reported_tracks
                ):
                    reported_tracks.add(pid)

                    frame_path = FRAMES_DIR / f"helmet_violation_track_{pid}_frame_{frame_idx}.jpg"
                    cv2.imwrite(str(frame_path), frame)

                    plate_text = ""
                    selected_plate_path = None

                    if plate_supported and plates:
                        best_plate = None
                        best_area = -1
                        for plate in plates:
                            x1, y1, x2, y2 = plate["box"]
                            area = (x2 - x1) * (y2 - y1)
                            if area > best_area:
                                best_area = area
                                best_plate = plate

                        if best_plate:
                            crop = clamp_crop(frame, best_plate["box"])
                            selected_plate_path = CROPS_DIR / f"plate_track_{pid}_frame_{frame_idx}.jpg"
                            cv2.imwrite(str(selected_plate_path), crop)
                            plate_text = plate_ocr(reader, crop)

                    record = {
                        "type": "helmet_violation_candidate",
                        "track_id": pid,
                        "frame_index": frame_idx,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "plate_text": plate_text,
                        "evidence_frame": str(frame_path),
                        "plate_crop": str(selected_plate_path) if selected_plate_path else None,
                        "review_required": True,
                    }
                    violation_log.append(record)
                    save_json(violation_log, LOG_JSON)

                    subject = f"Review Required: Possible Helmet Violation | Track {pid}"
                    body = (
                        f"Possible helmet violation detected.\n\n"
                        f"Track ID: {pid}\n"
                        f"Frame: {frame_idx}\n"
                        f"Timestamp (sec): {timestamp_sec:.2f}\n"
                        f"Detected plate: {plate_text or 'N/A'}\n"
                        f"Evidence frame: {frame_path}\n"
                        f"Plate crop: {selected_plate_path if selected_plate_path else 'N/A'}\n\n"
                        f"This alert is for human review only."
                    )
                    send_review_email(subject, body)

        for item in persons + helmets + vehicles + plates:
            x1, y1, x2, y2 = item["box"]
            label = f"{item['class_name']} | id={item['track_id']} | {item['conf']:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame, label, (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        cv2.imshow("Detection + Tracking + Review Alerts", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    save_json(violation_log, LOG_JSON)
    print(f"Saved violation log to: {LOG_JSON}")


if __name__ == "__main__":
    run_pipeline()