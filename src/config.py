# src/config.py
"""Centralized configuration — all secrets and tunables loaded from environment."""

import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _camera_source(value: str):
    try:
        return int(value)
    except ValueError:
        return value


def _project_path(value: str, default: Path) -> str:
    path = Path(value) if value else default
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path)


class Config:
    # ── Camera ──────────────────────────────────────────────────────────
    CAMERA_SOURCE = _camera_source(os.getenv("CAMERA_SOURCE", "0"))
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", "640"))
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", "480"))
    TARGET_FPS = int(os.getenv("TARGET_FPS", "15"))

    # ── Detection ───────────────────────────────────────────────────────
    FACE_DISTANCE_THRESHOLD = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.55"))
    HAZARD_CONSECUTIVE_FRAMES = int(os.getenv("HAZARD_CONSECUTIVE_FRAMES", "5"))
    FACE_ENCODINGS_PATH = _project_path(
        os.getenv("FACE_ENCODINGS_PATH", "models/face_encodings.pkl"),
        PROJECT_ROOT / "models" / "face_encodings.pkl",
    )
    YOLO_PERSON_MODEL = _project_path(
        os.getenv("YOLO_PERSON_MODEL", "yolov8n.pt"), PROJECT_ROOT / "yolov8n.pt"
    )
    YOLO_HAZARD_MODEL = _project_path(
        os.getenv("YOLO_HAZARD_MODEL", "yolov8m.pt"), PROJECT_ROOT / "yolov8m.pt"
    )

    # Per-class confidence overrides for hazard detection
    HAZARD_CONF_THRESHOLD = float(os.getenv("HAZARD_CONF_THRESHOLD", "0.6"))
    HAZARD_CLASS_THRESHOLDS = {
        "knife": 0.70,
        "scissors": 0.75,
        "gun": 0.60,
        "fire": 0.50,
        "smoke": 0.55,
    }

    # ── Notifications ───────────────────────────────────────────────────
    TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
    TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
    TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER")
    ADMIN_PHONE_NUMBERS = [
        n.strip() for n in os.getenv("ADMIN_PHONE_NUMBERS", "").split(",") if n.strip()
    ]

    SMTP_EMAIL = os.getenv("SMTP_EMAIL")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
    SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))

    # ── Dashboard ───────────────────────────────────────────────────────
    SECRET_KEY = os.getenv("SECRET_KEY", os.urandom(32).hex())
    ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH")
    ALLOW_INSECURE_DEFAULT_AUTH = os.getenv("ALLOW_INSECURE_DEFAULT_AUTH", "false").lower() == "true"
    SESSION_LIFETIME_HOURS = int(os.getenv("SESSION_LIFETIME_HOURS", "8"))

    # ── Database ────────────────────────────────────────────────────────
    DB_PATH = _project_path(os.getenv("DB_PATH", "data/eyenet.db"), PROJECT_ROOT / "data" / "eyenet.db")

    # ── Logging ─────────────────────────────────────────────────────────
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE = _project_path(os.getenv("LOG_FILE", "data/logs/eyenet.log"), PROJECT_ROOT / "data" / "logs" / "eyenet.log")
