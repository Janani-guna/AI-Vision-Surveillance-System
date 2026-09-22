# EyeNet – Real-Time Campus Surveillance

EyeNet is a **real-time computer vision surveillance system** that analyzes live camera feeds to detect security and compliance incidents and generate automated alerts.

## Features

* Face recognition and unknown-person detection
* YOLOv8-based hazard detection
* Uniform compliance detection
* ByteTrack-based object tracking
* Anomaly scoring and severity classification
* Real-time alerts via SMS and email
* Live monitoring dashboard
* Incident history and analytics

## Workflow

```text
Camera
  ↓
OpenCV Frame Capture
  ↓
Face + Uniform + YOLOv8 Detection
  ↓
Object Tracking
  ↓
Anomaly & Severity Analysis
  ↓
Incident Logging
  ↓
Dashboard + SMS/Email Alerts
```

## Tech Stack

**Python | OpenCV | YOLOv8 | PyTorch | NumPy | Flask | SQLite | ByteTrack | Twilio | Docker**

## Dashboard

* Live video streaming
* Real-time alerts
* Alert filtering and acknowledgement
* Incident snapshots
* FPS and performance metrics

## Installation

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
python -m src.encoders.encode_faces
python -m src.main
```

Dashboard: `http://localhost:5000`

## Docker

```bash
docker compose up --build
```

## Future Scope

* GPU/Edge AI optimization
* Role-based access control
* Scalable event processing
* Cloud storage and deployment
* Advanced alert management

## License

No license is currently included in the repository.
