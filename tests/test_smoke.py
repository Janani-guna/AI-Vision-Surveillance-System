import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.config import Config, PROJECT_ROOT
from src.core.database import close_db, get_latest_metrics, init_db, insert_metrics
from src.core.tracker import ObjectTracker
from src.dashboard.app import app
from src.dashboard.auth import hash_password


class EyeNetSmokeTests(unittest.TestCase):
    def test_configuration_uses_repository_paths(self):
        self.assertEqual(Path(Config.YOLO_HAZARD_MODEL), PROJECT_ROOT / "yolov8m.pt")
        self.assertEqual(Path(Config.FACE_ENCODINGS_PATH), PROJECT_ROOT / "models" / "face_encodings.pkl")

    def test_database_initializes_and_stores_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            database_path = os.path.join(directory, "eyenet.db")
            with patch.object(Config, "DB_PATH", database_path):
                init_db()
                insert_metrics(10.0, 2, 1, 25.0, 1, "online")
                metrics = get_latest_metrics()
                close_db()
        self.assertEqual(metrics["active_tracks"], 1)
        self.assertEqual(metrics["camera_status"], "online")

    def test_tracker_preserves_id_without_supervision(self):
        tracker = ObjectTracker()
        first = tracker.update([{"label": "hazard:knife", "conf": 0.9, "box": (10, 10, 50, 50)}], 1)
        second = tracker.update([{"label": "hazard:knife", "conf": 0.9, "box": (11, 11, 51, 51)}], 2)
        self.assertEqual(first[0].track_id, second[0].track_id)

    def test_login_and_protected_route(self):
        client = app.test_client()
        password_hash = hash_password("test-password")
        with patch.object(Config, "ADMIN_PASSWORD_HASH", password_hash):
            response = client.post("/login", data={"username": "admin", "password": "test-password"})
            self.assertEqual(response.status_code, 302)
            self.assertEqual(client.get("/").status_code, 200)
        self.assertEqual(app.test_client().get("/video_feed").status_code, 302)


if __name__ == "__main__":
    unittest.main()