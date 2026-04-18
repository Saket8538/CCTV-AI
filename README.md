# CCTV AI — Missing Person Detection System

A computer-vision application to register missing-person cases, scan live CCTV/video footage, match detected faces against registered references, and help human operators verify and notify complainants.

This repository contains two frontends over a shared detection core:
- **Streamlit web app** (`streamlit_app.py`) — primary modern interface.
- **Tkinter desktop app** (`app.py`) — legacy Windows-focused interface.

The face-matching engine is implemented in **`enhanced_detector.py`** using InsightFace embeddings, quality filtering, voting logic, and optional tracking/refinement for CCTV footage.

---

## 1) Key Features

- Register missing-person cases with profile details and one/multiple reference photos.
- Store case records in SQLite (`Database/data.db`).
- Process:
  - **Live camera stream** (camera index)
  - **Uploaded CCTV footage files** (`.mp4`, `.avi`, `.mkv`, `.mov`)
- Face recognition against all active cases (`status IN (0,1)`).
- Detection artifact storage (full frame + face crop) under `found/<pid>/<source>/...`.
- Case lifecycle management:
  - Not detected
  - Pending verification
  - Verified
  - Failed / not a match
- Optional notification channels on confirmation:
  - Email (SMTP)
  - SMS (Twilio)
- Reference-image management (add/remove extra refs) in Streamlit UI.

---

## 2) Repository Structure

```text
CCTV-AI/
├── streamlit_app.py          # Main web UI and orchestration
├── enhanced_detector.py      # Core detection + matching engine
├── app.py                    # Legacy Tkinter UI (Windows-oriented)
├── requirements.txt          # Python dependencies
├── yolov8n.pt                # YOLO model used by Streamlit live pipeline
├── Database/
│   └── data.db               # SQLite database
├── data/                     # Stored reference images for registered cases
├── found/                    # Output detections grouped by person/source
├── logs/                     # Runtime logs (created at runtime)
├── temp/                     # Temporary uploaded videos (created at runtime)
├── scan_info.json            # Detection state/pending verification map
├── .streamlit/
│   └── secrets.toml          # Local secrets (not for public sharing)
└── creds.json                # Legacy credentials fallback (Twilio)
```

---

## 3) Core Architecture

### 3.1 Streamlit app (`streamlit_app.py`)
Main responsibilities:
- Initializes/migrates DB schema (`init_database`).
- Handles registration form + image preprocessing.
- Runs live detection display pipeline (`PersonDetector` + `FaceDetector`).
- Runs uploaded-footage processing (`process_video_file`) by calling `MissingPersonDetector`.
- Provides menus for dashboard, registration, live detection, footage processing, reference management, verification, and search.
- Sends notifications when operator confirms a match.

### 3.2 Detection engine (`enhanced_detector.py`)
`MissingPersonDetector` does:
- Loads InsightFace (`buffalo_l`) with GPU provider when available.
- Rebuilds embedding gallery from all active case reference images.
- Reads live/video frames and detects faces.
- Applies face quality checks (size, blur, brightness).
- Computes cosine similarity against gallery embeddings.
- Uses score threshold + confidence margin for safer matches.
- Uses vote window (`surity`) and frame time gap to reduce duplicate writes.
- Persists detections and updates `scan_info.json` + DB pending status.
- For footage mode, uses lightweight tracking (`_FootageTracker`) and optional focused refine pass.

### 3.3 Legacy desktop app (`app.py`)
- Tkinter-based case management UI.
- Calls `MissingPersonDetector` for live/footage detection.
- Includes match/no-match actions and notification logic.
- Uses Windows-specific modules (`win32api`) and is primarily for Windows environments.

---

## 4) Database Model

Table: **`missing_people`**

Important columns:
- `id` (PK)
- Person details: `name`, `gender`, `age`, `missing_state`, `missing_city`, `pincode`, `missing_date`, `description`
- Complainant details: `complaint_name`, `complaint_phone`, `complaint_email`, `complaint_address`
- Media: `image_f` (primary), `reference_images_json` (list), `reference_count`, `footage_path`
- Detection state: `status`
- Notification tracking: `notification_email_sent`, `notification_sms_sent`, `notification_last_message`
- Metadata: `face_encoding`, `created_at`

### Status values used in code
- `0` → Not detected
- `1` → Pending verification
- `2` → Verified/confirmed
- `3` → Failed / not a match

---

## 5) Detection & Matching Flow

1. **Case registration**
   - Save person + complainant metadata in DB.
   - Save primary and optional extra references in `data/`.
   - Build/update references list in `reference_images_json`.

2. **Face gallery build**
   - On detection start, active cases are loaded.
   - Embeddings extracted from each available reference image.

3. **Frame processing**
   - Detect faces in frame.
   - Filter low-quality faces.
   - Match embeddings against gallery.
   - For footage, tracking + re-identification helps stability.

4. **Vote and store**
   - Require repeated confidence (`surity`) before recording.
   - Save `*_full.jpg` and `*_face.jpg` into `found/`.
   - Move case to pending status (`status=1`).

5. **Human verification**
   - Operator confirms/rejects in UI.
   - On confirm: status set to `2` and notifications attempted.

---

## 6) Installation

> Recommended: Python 3.10+ in a virtual environment.

```bash
cd /home/runner/work/CCTV-AI/CCTV-AI
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

Dependencies include Streamlit, OpenCV, Ultralytics, InsightFace, ONNX Runtime, Twilio, and support libraries.

---

## 7) Configuration

### 7.1 Notifications via Streamlit secrets
Create/update `.streamlit/secrets.toml`:

```toml
[email]
sender = "your-email@example.com"
password = "app-password-or-smtp-password"
to = "team-alerts@example.com"
smtp_host = "smtp.gmail.com"
smtp_port = 587

[twilio]
sid = "ACxxxxxxxx"
token = "xxxxxxxx"
from_ = "+1xxxxxxxxxx"  # or use messaging_service_sid
# messaging_service_sid = "MGxxxxxxxx"
```

`app.py` can also fallback to Twilio credentials from `creds.json`.

---

## 8) Running the Project

### 8.1 Run Streamlit app (primary)
```bash
cd /home/runner/work/CCTV-AI/CCTV-AI
streamlit run streamlit_app.py
```

### 8.2 Run detector directly (CLI)
```bash
# Process footage
python enhanced_detector.py --footage_path /absolute/path/to/video.mp4

# Live camera
python enhanced_detector.py --cam_id 0
```

Useful CLI knobs:
- `--surity`
- `--frame_time_gap`
- `--threshold`
- `--process_every_n_frames`
- `--min_face_size`
- `--blur_threshold`

### 8.3 Run legacy Tkinter app (Windows-oriented)
```bash
python app.py
```

---

## 9) Streamlit UI Modules

- **Dashboard**: totals and recent records.
- **Register Missing Person**: create case + upload references + optional footage.
- **Live Detection**: webcam-based real-time detection display.
- **Process Footage**: upload video, auto-profile selection (Fast/Balanced/Accurate), run detection.
- **Manage References**: inspect/add/remove references per person.
- **Verify Matches**: review pending detections, confirm/reject, send notifications.
- **Search Records**: filter by name/location/status and inspect history.

---

## 10) Output & Artifacts

- **Reference images**: `data/<pid>_*.jpg`
- **Detections**: `found/<pid>/<source>/<timestamp>_full.jpg` and `_face.jpg`
- **State file**: `scan_info.json` keeps pending PID mappings + detector errors
- **Logs**:
  - `logs/streamlit_app.log`
  - `logs/detection.log`
  - `logs/app.log` (legacy)

---

## 11) Operational Notes

- GPU acceleration is used automatically if `CUDAExecutionProvider` is available; otherwise CPU is used.
- Streamlit and detector code create missing runtime folders automatically.
- The detector only matches against active cases (`status 0 or 1`).
- Footage processing includes orientation probing and optional focused refinement for harder frames.

---

## 12) Troubleshooting

- **`No module named pytest`**: tests are not configured in this repository by default.
- **`Failed to open video source`**: check camera index/file path and codec support.
- **No matches detected**:
  - Add more clear reference images.
  - Improve lighting/face size in footage.
  - Try lower threshold / stricter processing profile.
- **Notifications not sent**:
  - Verify `.streamlit/secrets.toml` keys.
  - Check SMTP/Twilio credentials and network access.
- **Streamlit missing assets error** (`assets/...`): ensure required image assets exist in expected paths.

---

## 13) Security & Privacy Guidance

- Do **not** commit real credentials in `creds.json` or `.streamlit/secrets.toml`.
- Treat stored person images and detections as sensitive data.
- Restrict access to DB, logs, and artifact folders.
- Configure secure retention/deletion policies for personal data.

---

## 14) License

This project includes an MIT License (`LICENSE`).
