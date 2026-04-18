# CCTV AI - Missing Person Detection System

A Windows-first, Python-based system for missing-person search across live camera feeds and uploaded CCTV footage.

This project includes:

- A Streamlit web application for case registration, live detection, footage processing, verification, and search.
- A Tkinter desktop application for similar workflows in a desktop GUI.
- A shared high-accuracy detector based on InsightFace embeddings with tracking and vote-based confirmation.
- SQLite-backed case management and local filesystem storage for references and detections.

## Key Features

- Missing person case registration with one primary + optional multiple reference images.
- Live camera detection (web + desktop modes).
- Uploaded CCTV footage processing with auto-tuned quality profile.
- Face matching using cosine similarity over InsightFace embeddings.
- Quality gating (min face size, blur threshold, brightness checks).
- Multi-frame voting (`surity`) to reduce false positives.
- Optional focused refine pass for hard CCTV frames.
- Match verification flow with email and Twilio SMS notifications.
- Search, filtering, and deletion of case records.

## Tech Stack

- Python 3.10+ (project has a local `.venv`)
- Streamlit UI
- Tkinter UI
- OpenCV
- InsightFace (`buffalo_l` model via `FaceAnalysis`)
- ONNX Runtime (CPU/GPU providers)
- YOLOv8 Nano (`yolov8n.pt`) in live Streamlit person-first detection mode
- SQLite (`Database/data.db`)
- Pandas, Pillow, Loguru
- Twilio (optional notifications)
- SMTP email (optional notifications)

## Project Structure

- `streamlit_app.py`: Main web app (recommended operational UI).
- `app.py`: Tkinter desktop GUI (legacy/desktop flow).
- `enhanced_detector.py`: Shared detector engine and CLI entry point.
- `requirements.txt`: Python dependencies.
- `Database/data.db`: SQLite database.
- `data/`: Registered reference images for each person.
- `found/`: Detection outputs (`*_face.jpg`, `*_full.jpg`) grouped by person/source.
- `scan_info.json`: Shared pending-verification state and detector errors.
- `logs/`: Runtime logs (`detection.log`, `streamlit_app.log`, optionally `app.log`).
- `.streamlit/secrets.toml`: Notification and credential config for Streamlit + desktop fallback logic.
- `creds.json`: Legacy Twilio credential fallback for desktop app.

## Detection Architecture

### 1) Gallery Construction

At startup and during sync points, the detector builds an embedding gallery from active records (`status IN (0,1)`) by reading:

- `image_f`
- `reference_images_json`
- matching files like `data/{pid}_*.jpg`

For each image, the largest detected face embedding is normalized and stored under that PID.

### 2) Frame Processing

For each processed frame:

- Faces are detected via InsightFace.
- Optional multiscale retry runs for difficult footage.
- Quality checks filter poor candidates:
  - minimum face size
  - blur threshold (Laplacian variance)
  - brightness sanity bounds
- Embeddings are matched using cosine similarity against gallery references.
- Matching requires:
  - score >= `similarity_threshold`
  - margin over second-best >= `confidence_margin`

### 3) Temporal Stabilization

To reduce false positives, detections are only persisted after repeated hits:

- `surity` controls votes needed within `vote_window_seconds`.
- `frame_time_gap` limits save frequency for the same PID.
- Footage mode uses lightweight IOU+Kalman tracking (`_FootageTracker`) before re-identification intervals.

### 4) Persistence and Verification

Confirmed detections are saved to:

- `found/{pid}/{source}/..._face.jpg`
- `found/{pid}/{source}/..._full.jpg`

`scan_info.json` receives pending paths under `for_verification_pids`, and DB status is moved from `0` to `1` for pending verification.

## Database Schema

Table: `missing_people`

Core columns used by current code:

- `id` INTEGER PRIMARY KEY AUTOINCREMENT
- `name` TEXT
- `gender` TEXT
- `age` INTEGER
- `missing_state` TEXT
- `missing_city` TEXT
- `pincode` INTEGER
- `missing_date` TEXT
- `description` TEXT
- `image_f` TEXT
- `complaint_name` TEXT
- `complaint_phone` TEXT
- `complaint_email` TEXT
- `complaint_address` TEXT
- `footage_path` TEXT
- `status` INTEGER (0 not detected, 1 pending, 2 verified, 3 failed)
- `notification_email_sent` INTEGER
- `notification_sms_sent` INTEGER
- `notification_last_message` TEXT
- `face_encoding` TEXT
- `created_at` TIMESTAMP
- `reference_images_json` TEXT
- `reference_count` INTEGER

The Streamlit app includes safe migration logic to add missing columns when needed.

## Prerequisites

- Windows machine (desktop app relies on `pywin32` monitor APIs).
- A working webcam or camera stream for live mode.
- Python with local virtual environment at `.venv`.
- Optional for notifications:
  - SMTP credentials (Gmail/app password or equivalent)
  - Twilio account credentials and sender configuration

## Setup

1. Open terminal in project root.
2. Activate local virtual environment.
3. Install dependencies.
4. Confirm model file `yolov8n.pt` exists in root.

### Windows Command Prompt

```bat
cd c:\Users\Desktop\CCTV-AI
.venv\Scripts\activate
pip install -r requirements.txt
```

### PowerShell

```powershell
Set-Location c:\Users\Desktop\CCTV-AI
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Configuration

### Preferred: `.streamlit/secrets.toml`

```toml
[email]
sender = "your_email@example.com"
password = "your_app_password"
to = "team_or_default_recipient@example.com"
smtp_host = "smtp.gmail.com"
smtp_port = 587

[twilio]
sid = "ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
token = "your_auth_token"
from_ = "+1xxxxxxxxxx"
# or
messaging_service_sid = "MGxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```

### Legacy fallback: `creds.json`

Desktop flow can fallback to legacy Twilio values from:

```json
{
  "twilio": {
    "sid": "...",
    "token": "...",
    "from_": "+1...",
    "messaging_service_sid": "MG..."
  }
}
```

## Run the Applications

### Streamlit (Recommended)

```bat
cd c:\Users\Desktop\CCTV-AI
.venv\Scripts\activate
streamlit run streamlit_app.py
```

Then open the shown local URL in browser.

Sidebar modules:

- Dashboard
- Register Missing Person
- Live Detection
- Process Footage
- Manage References
- Verify Matches
- Search Records

### Tkinter Desktop App

```bat
cd c:\Users\Desktop\CCTV-AI
.venv\Scripts\activate
python app.py
```

## CLI Detector Usage

`enhanced_detector.py` provides a direct CLI mode for scripted runs.

### Live camera

```bat
cd c:\Users\Desktop\CCTV-AI
.venv\Scripts\activate
python enhanced_detector.py --cam_id 0
```

### Footage file

```bat
cd c:\Users\Desktop\CCTV-AI
.venv\Scripts\activate
python enhanced_detector.py --footage_path "temp\sample.mp4"
```

### Useful detector flags

- `--surity` votes required before persisting a match.
- `--frame_time_gap` minimum seconds between saved detections for same PID.
- `--threshold` cosine similarity threshold.
- `--process_every_n_frames` skip factor for performance.
- `--min_face_size` reject tiny faces.
- `--blur_threshold` reject blurry faces.

## Typical Workflow

1. Register missing person details and at least one clear face image.
2. Optionally add more reference images to improve matching.
3. Start live detection or process uploaded footage.
4. Review pending matches (`status = 1`).
5. Confirm or reject:
   - Confirm -> `status = 2` and notifications attempted.
   - Reject -> `status = 3`.
6. Use Search for historical case lookup and cleanup.

## Important Files and Data Contracts

### `scan_info.json`

Expected shape:

```json
{
  "for_verification_pids": {
    "16": ["found/16/source_x/20260406T095726_face.jpg"]
  },
  "error": ""
}
```

### Output Paths

- Registered references: `data/{pid}_*.jpg`
- Detections: `found/{pid}/{source}/{timestamp}_face.jpg` and `_full.jpg`
- Logs: `logs/detection.log`, `logs/streamlit_app.log`

## Performance and Accuracy Notes

- Better reference quality and variety improve matching significantly.
- For low-light CCTV, keep enhancement enabled and provide multiple references.
- `process_every_n_frames` is the main speed knob.
- Lower `threshold` increases recall but can increase false positives.
- Higher `surity` reduces false positives but may miss short appearances.

## Known Operational Notes

- Desktop app assumes image assets such as `assets/main_pg.png`, `assets/complaint_pg.png`, and `assets/refresh.png` are available. If missing, startup will fail.
- Desktop app insert query is legacy-oriented and may diverge from newer schema expectations maintained by Streamlit migrations.
- Streamlit is the most up-to-date and complete workflow in this codebase.

## Security Checklist

- Do not commit real credentials to source control.
- Add local secret files to `.gitignore` (`.streamlit/secrets.toml`, `creds.json`).
- Rotate any exposed Twilio or email credentials immediately.
- Prefer environment variables or secret managers for production deployment.

## Troubleshooting

### 1) `No face detected` during registration

- Use a frontal, well-lit image.
- Increase image quality/resolution.
- Enable low-light enhancement in Streamlit registration.

### 2) Camera fails in live mode

- Verify camera index (`0`, `1`, etc.).
- Close other apps using webcam.
- Confirm OpenCV camera permissions in Windows.

### 3) Slow footage processing

- Use faster profile defaults (higher frame skipping).
- Reduce video resolution/duration before upload.
- Ensure GPU ONNX Runtime is usable if CUDA is installed.

### 4) Notifications not sent

- Validate `.streamlit/secrets.toml` values.
- For email: check SMTP host, TLS, app password.
- For Twilio: check SID/token and either `from_` or `messaging_service_sid`.

### 5) Pending matches not visible

- Check `scan_info.json` for detector errors.
- Confirm record status remains in active states (`0`/`1`) during scans.
- Verify files are being written to `found/`.

## Development Notes

- Existing environment appears to include `.venv`; keep all installs local to it.
- Logging is file-based; monitor `logs/` during long detection runs.
- Detector code in `enhanced_detector.py` is shared by both UI entry points.

## License

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

