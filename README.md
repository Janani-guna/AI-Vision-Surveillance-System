# EyeNet – Real-Time Campus Surveillance & Incident Response

EyeNet is a **real-time computer vision surveillance system** designed for campus and institutional environments. It processes live camera feeds, detects security and compliance events, converts them into structured incidents, and provides real-time monitoring and automated notifications.

## Key Features

* Real-time face recognition and unknown-person detection
* YOLOv8-based hazard detection for objects such as weapons, fire, and smoke
* Uniform compliance detection using computer vision
* Cross-frame object tracking using ByteTrack
* Temporal persistence to reduce false alerts
* Anomaly scoring from detection confidence and track persistence
* Severity-based incident prioritization
* Automated SMS and email notifications
* Real-time dashboard with live video and alerts
* Alert history, filtering, acknowledgement, and analytics
* SQLite database with WAL support for operational data

## System Workflow

```text
Camera / Video Feed
        ↓
Frame Capture
        ↓
Computer Vision Detection
 ┌────────┼───────────┐
 ↓        ↓           ↓
Face   Uniform      YOLOv8
Recognition Check    Hazard Detection
 └────────┼───────────┘
          ↓
     Object Tracking
          ↓
   Anomaly & Severity
       Analysis
          ↓
     Incident Event
      ┌───┼────┐
      ↓   ↓    ↓
   SQLite SMS  Email
      ↓
 Web Dashboard
```

## Tech Stack

**Programming:** Python
**Computer Vision:** OpenCV, face-recognition
**Object Detection:** Ultralytics YOLOv8
**Object Tracking:** ByteTrack / IoU-based fallback
**ML:** PyTorch, NumPy
**Backend & Dashboard:** Flask, REST API, SSE
**Database:** SQLite
**Notifications:** Twilio SMS, SMTP Email
**Security:** bcrypt, Flask Sessions
**Deployment:** Docker, Docker Compose

## Architecture

The system uses a shared camera buffer so multiple components can safely process the same video stream. Detection results are passed through an object tracker and converted into prioritized events. An event bus then distributes incidents to the database, notification services, and live dashboard.

## Detection & Alert System

EyeNet assigns severity levels based on detected events:

| Event                       | Severity |
| --------------------------- | -------- |
| Gun / Fire / Explosive      | Critical |
| Knife / Smoke / Axe / Sword | High     |
| Unknown Person              | Medium   |
| Uniform Violation           | Low      |

Anomaly scores range from **0–100** and consider event type, detection confidence, and track persistence.

## Dashboard

The Flask dashboard provides:

* Secure admin login
* Live camera streaming
* Real-time alerts using Server-Sent Events
* Alert filtering and acknowledgement
* Incident snapshots
* FPS and processing metrics
* Hourly incident analytics

## Database

SQLite stores:

* Incident/alert records
* Detection metadata
* Track IDs
* Snapshots
* Pipeline performance metrics
* Student enrollment metadata

WAL mode is enabled to improve concurrent dashboard reads and database writes.

## Performance Optimizations

* Shared camera frame buffer
* Downscaled face recognition
* Temporal hazard persistence
* Tracking-based alert confirmation
* Notification cooldowns
* SQLite WAL mode
* Rotating application logs

## Installation

### Prerequisites

* Python 3.11 recommended
* OpenCV-compatible camera
* Optional Twilio account for SMS
* Optional SMTP credentials for email

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Configure the required environment variables using `.env`:

```bash
Copy-Item .env.example .env
```

Generate face encodings:

```bash
python -m src.encoders.encode_faces
```

Start the application:

```bash
python -m src.main
```

Dashboard:

```text
http://localhost:5000
```

## Docker Deployment

```bash
docker compose up --build
```

Docker Compose supports the backend, dashboard, database, and camera passthrough configuration. Camera passthrough is currently Linux-specific.

## Testing

```bash
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
```

The tests cover configuration, database initialization, metrics, authentication, dashboard routes, and tracking fallback.

## Future Improvements

* GPU/edge accelerator support
* Redis-based event messaging for multi-process scaling
* Role-based access control
* API rate limiting
* Improved alert escalation workflow
* GPU model warmup and inference optimization
* Cloud/object-storage support for incident snapshots

## License

No license file is currently included. Add an appropriate open-source license before redistributing the project.
