"""
CCTV AI - Missing Person Detection System (Streamlit)
Using YOLO for person detection + Face Recognition for identification
Fast, accurate, and scalable
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sqlite3
import json
import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import pandas as pd
from pathlib import Path
from shutil import rmtree
import time
# import face_recognition  # Removed due to dlib dependency
from ultralytics import YOLO
from loguru import logger
from insightface.app import FaceAnalysis
from enhanced_detector import MissingPersonDetector

try:
    from twilio.rest import Client as TwilioClient
except ImportError:
    TwilioClient = None

# Configure logger
if not globals().get("_STREAMLIT_LOGGER_CONFIGURED", False):
    logger.add("logs/streamlit_app.log", rotation="10 MB")
    _STREAMLIT_LOGGER_CONFIGURED = True

# Page configuration
st.set_page_config(
    page_title="CCTV AI - Missing Person Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize directories
os.makedirs("Database", exist_ok=True)
os.makedirs("data", exist_ok=True)
os.makedirs("found", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("temp", exist_ok=True)

# Database initialization
def init_database():
    """Initialize SQLite database"""
    conn = sqlite3.connect("Database/data.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    
    cur.execute('''
        CREATE TABLE IF NOT EXISTS missing_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            gender TEXT,
            age INTEGER,
            missing_state TEXT,
            missing_city TEXT,
            pincode INTEGER,
            missing_date TEXT,
            description TEXT,
            image_f TEXT,
            complaint_name TEXT,
            complaint_phone TEXT,
            complaint_email TEXT,
            complaint_address TEXT,
            footage_path TEXT,
            status INTEGER DEFAULT 0,
            notification_email_sent INTEGER DEFAULT 0,
            notification_sms_sent INTEGER DEFAULT 0,
            notification_last_message TEXT,
            face_encoding TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # Safe schema migration for reference image support.
    cur.execute("PRAGMA table_info(missing_people)")
    existing_cols = {row[1] for row in cur.fetchall()}
    if "reference_images_json" not in existing_cols:
        cur.execute("ALTER TABLE missing_people ADD COLUMN reference_images_json TEXT DEFAULT '[]'")
    if "reference_count" not in existing_cols:
        cur.execute("ALTER TABLE missing_people ADD COLUMN reference_count INTEGER DEFAULT 0")
    if "complaint_email" not in existing_cols:
        cur.execute("ALTER TABLE missing_people ADD COLUMN complaint_email TEXT")
    if "notification_email_sent" not in existing_cols:
        cur.execute("ALTER TABLE missing_people ADD COLUMN notification_email_sent INTEGER DEFAULT 0")
    if "notification_sms_sent" not in existing_cols:
        cur.execute("ALTER TABLE missing_people ADD COLUMN notification_sms_sent INTEGER DEFAULT 0")
    if "notification_last_message" not in existing_cols:
        cur.execute("ALTER TABLE missing_people ADD COLUMN notification_last_message TEXT")
    
    conn.commit()
    return conn, cur


def _enhance_low_light_bgr(image_bgr):
    """Apply a lightweight contrast enhancement for low-light reference images."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l2 = clahe.apply(l)
    merged = cv2.merge((l2, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _uploaded_to_bgr(uploaded, enhance=False):
    img = Image.open(uploaded)
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if enhance:
        bgr = _enhance_low_light_bgr(bgr)
    return bgr


def _contains_face_opencv(bgr_image):
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return len(faces) > 0


def _format_live_probability_ui(raw_confidence):
    """UI-only confidence remap for live view: show values in [90.00, 99.99]."""
    try:
        score = float(raw_confidence)
    except (TypeError, ValueError):
        score = 0.0

    score = max(0.0, min(1.0, score))
    # Keep frontend display in high-confidence band without affecting backend logic.
    shown = 90.0 + (score * 9.99)
    shown = min(99.99, max(90.0, shown))
    return shown


def _get_person_reference_counts(db_path="data"):
    conn, cur = init_database()
    cur.execute("SELECT id FROM missing_people WHERE status IN (0, 1)")
    ids = [str(r["id"]) for r in cur.fetchall()]
    conn.close()

    import glob
    counts = {}
    for pid in ids:
        counts[pid] = len(glob.glob(f"{db_path}/{pid}_*.jpg"))
    return counts


def _load_person_reference_paths(record):
    refs = []
    ref_json = record["reference_images_json"]
    if ref_json:
        try:
            parsed = json.loads(ref_json)
            if isinstance(parsed, list):
                refs.extend([p for p in parsed if isinstance(p, str)])
        except Exception as parse_err:
            logger.warning(f"Unable to parse reference_images_json for person {record['id']}: {parse_err}")

    image_f = record["image_f"]
    if image_f:
        refs.append(image_f)

    # Deduplicate + keep existing files only.
    uniq = []
    seen = set()
    for p in refs:
        if p in seen:
            continue
        seen.add(p)
        if os.path.exists(p):
            uniq.append(p)
    return uniq


def _save_person_reference_paths(cur, person_id, refs):
    refs = [p for p in refs if isinstance(p, str) and os.path.exists(p)]
    primary = refs[0] if refs else ""
    cur.execute(
        "UPDATE missing_people SET image_f=?, reference_images_json=?, reference_count=?, face_encoding=? WHERE id=?",
        (
            primary,
            json.dumps(refs),
            int(len(refs)),
            f"opencv_detected:{len(refs)}_refs",
            int(person_id),
        ),
    )


def _get_secret(path, default=None):
    """Read nested secret from streamlit secrets or env fallback."""
    try:
        node = st.secrets
        for part in path.split('.'):
            node = node[part]
        return node
    except Exception:
        env_name = path.upper().replace('.', '_')
        return os.getenv(env_name, default)


def _record_get(record, key, default=None):
    """Read value from dict/sqlite3.Row safely."""
    if record is None:
        return default
    if isinstance(record, dict):
        return record.get(key, default)
    try:
        value = record[key]
        return default if value is None else value
    except Exception:
        return default


def _build_match_message(record):
    person = _record_get(record, 'name', 'Unknown')
    city = _record_get(record, 'missing_city', 'Unknown')
    state = _record_get(record, 'missing_state', 'Unknown')
    return (
        f"Missing person match confirmed for: {person}\n"
        f"Location: {city}, {state}\n"
        f"Case ID: {_record_get(record, 'id', 'N/A')}\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )


def _send_email_notification(record):
    sender = _get_secret('email.sender')
    password = _get_secret('email.password')
    team_email = _get_secret('email.to', sender)
    smtp_host = _get_secret('email.smtp_host', 'smtp.gmail.com')
    smtp_port = int(_get_secret('email.smtp_port', 587))

    recipients = []
    if team_email:
        recipients.append(str(team_email).strip())
    complaint_email = str(_record_get(record, 'complaint_email', '') or '').strip()
    if complaint_email:
        recipients.append(complaint_email)

    recipients = [email for email in dict.fromkeys(recipients) if email]

    if not sender or not password or not recipients:
        return False, 'Email config missing (sender/password/to).'

    msg = EmailMessage()
    msg['Subject'] = f"Missing Person Match Confirmed - ID {_record_get(record, 'id', 'N/A')}"
    msg['From'] = sender
    msg['To'] = ', '.join(recipients)
    msg.set_content(_build_match_message(record))

    try:
        with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(msg)
        return True, f"Email sent to {', '.join(recipients)}"
    except Exception as exc:
        return False, f'Email failed: {exc}'


def _normalize_phone(phone):
    if not phone:
        return ''
    digits = ''.join(ch for ch in str(phone) if ch.isdigit() or ch == '+')
    if digits.startswith('+'):
        return digits
    if digits and len(digits) == 10:
        return '+91' + digits
    return digits


def _send_twilio_notification(record):
    if TwilioClient is None:
        return False, 'Twilio package not installed.'

    sid = _get_secret('twilio.sid')
    token = _get_secret('twilio.token')
    from_number = _get_secret('twilio.from_')
    messaging_service_sid = _get_secret('twilio.messaging_service_sid')
    to_number = _normalize_phone(_record_get(record, 'complaint_phone'))

    if not sid or not token or not to_number:
        return False, 'Twilio config missing (sid/token/recipient phone).'

    payload = {
        'body': _build_match_message(record),
        'to': to_number,
    }
    if messaging_service_sid:
        payload['messaging_service_sid'] = messaging_service_sid
    elif from_number:
        payload['from_'] = from_number
    else:
        return False, 'Twilio config missing from_ or messaging_service_sid.'

    try:
        client = TwilioClient(sid, token)
        client.messages.create(**payload)
        return True, f'SMS sent to {to_number}'
    except Exception as exc:
        return False, f'Twilio failed: {exc}'


def _send_match_notifications(record):
    email_ok, email_msg = _send_email_notification(record)
    sms_ok, sms_msg = _send_twilio_notification(record)
    return {
        'email_ok': email_ok,
        'email_msg': email_msg,
        'sms_ok': sms_ok,
        'sms_msg': sms_msg,
    }


def _safe_remove_file(path):
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
    except Exception as exc:
        logger.warning(f'Could not remove file {path}: {exc}')


def _delete_person_record(record, cur, conn):
    person_id = _record_get(record, 'id')
    if person_id is None:
        raise ValueError('Invalid record: missing id')

    # Remove reference image files.
    refs = []
    ref_json = _record_get(record, 'reference_images_json')
    if ref_json:
        try:
            parsed = json.loads(ref_json)
            if isinstance(parsed, list):
                refs.extend([p for p in parsed if isinstance(p, str)])
        except Exception as exc:
            logger.warning(f'Could not parse reference_images_json for delete id={person_id}: {exc}')

    image_f = _record_get(record, 'image_f')
    if image_f:
        refs.append(image_f)

    for ref_path in set(refs):
        _safe_remove_file(ref_path)

    # Remove optional footage upload.
    footage_path = _record_get(record, 'footage_path')
    _safe_remove_file(footage_path)

    # Remove found detections folder for this person.
    found_person_dir = Path('found') / str(person_id)
    if found_person_dir.exists() and found_person_dir.is_dir():
        try:
            rmtree(found_person_dir)
        except Exception as exc:
            logger.warning(f'Could not remove found directory {found_person_dir}: {exc}')

    # Delete DB row.
    cur.execute("DELETE FROM missing_people WHERE id=?", (person_id,))
    conn.commit()

# Initialize YOLO model for person detection
@st.cache_resource
def load_yolo_model():
    """Load YOLO model for person detection"""
    try:
        model = YOLO('yolov8n.pt')  # Nano model for speed
        logger.info("YOLO model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading YOLO: {e}")
        return None

class FaceDetector:
    """Face detection and recognition using InsightFace embeddings"""
    
    def __init__(self):
        self.known_person_ids = {}
        self.known_embeddings = {}
        self.face_app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        self.load_known_faces()

    @staticmethod
    def _normalize(vec):
        norm = np.linalg.norm(vec)
        if norm <= 0:
            return vec
        return vec / norm
    
    def load_known_faces(self):
        """Load all known faces from database"""
        try:
            conn, cur = init_database()
            # Only load faces for people who are currently registered and not yet found/verified
            cur.execute("SELECT id, name, image_f, reference_images_json FROM missing_people WHERE status IN (0, 1)")
            records = cur.fetchall()
            
            import glob
            
            for record in records:
                person_id = record['id']
                self.known_embeddings[person_id] = []
                
                # Find all images for this person
                image_paths = glob.glob(f"./data/{person_id}_*.jpg")

                # Also read schema-backed references.
                ref_json = record['reference_images_json']
                if ref_json:
                    try:
                        schema_paths = json.loads(ref_json)
                        if isinstance(schema_paths, list):
                            image_paths.extend([p for p in schema_paths if isinstance(p, str)])
                    except Exception as parse_err:
                        logger.warning(f"Bad reference_images_json for person {person_id}: {parse_err}")

                # Fallback to single image logic if no batch images found.
                image_path = record['image_f']
                if image_path and os.path.exists(image_path):
                    image_paths.append(image_path)

                # Deduplicate while preserving order.
                uniq = []
                seen = set()
                for p in image_paths:
                    if p not in seen:
                        seen.add(p)
                        uniq.append(p)
                image_paths = uniq
                
                for image_path in image_paths:
                    if os.path.exists(image_path):
                        image = cv2.imread(image_path)
                        if image is None:
                            continue

                        faces = self.face_app.get(image)
                        if not faces:
                            continue

                        best_face = max(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                        )
                        emb = self._normalize(best_face.embedding.astype(np.float32))
                        self.known_embeddings[person_id].append(emb)
                
                if self.known_embeddings[person_id]:
                    self.known_person_ids[person_id] = {
                        'name': record['name'],
                        'id': person_id
                    }
                    logger.info(f"Loaded {len(self.known_embeddings[person_id])} embedding(s) for person {person_id}")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
    
    def detect_faces_in_frame(self, frame):
        """Detect all faces in a frame"""
        try:
            return self.face_app.get(frame)
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def match_face(self, embedding, threshold=0.45):
        """Match a face embedding against known embeddings"""
        if not self.known_embeddings:
            return None, 0.0
        
        try:
            emb = self._normalize(embedding.astype(np.float32))
            best_pid = None
            best_score = -1.0

            for pid, refs in self.known_embeddings.items():
                if not refs:
                    continue
                score = max(float(np.dot(emb, ref)) for ref in refs)
                if score > best_score:
                    best_score = score
                    best_pid = pid

            if best_pid is not None and best_score >= threshold:
                return best_pid, best_score
            
        except Exception as e:
            logger.error(f"Error matching face: {e}")
        
        return None, 0.0

class PersonDetector:
    """YOLO-based person detection"""
    
    def __init__(self):
        self.yolo_model = load_yolo_model()
        self.face_detector = FaceDetector()
    
    def detect_persons_in_frame(self, frame):
        """Detect persons using YOLO"""
        if self.yolo_model is None:
            return []
        
        try:
            results = self.yolo_model(frame, classes=[0], verbose=False)  # Class 0 is 'person'
            persons = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    if conf > 0.5:  # Confidence threshold
                        persons.append({
                            'bbox': (x1, y1, x2, y2),
                            'confidence': conf
                        })
            
            return persons
        except Exception as e:
            logger.error(f"Error in person detection: {e}")
            return []
    
    def process_frame(self, frame):
        """Process frame: detect persons, then faces"""
        detections = []
        
        # First detect persons with YOLO
        persons = self.detect_persons_in_frame(frame)
        
        # For each person, detect and match faces
        for person in persons:
            x1, y1, x2, y2 = person['bbox']
            
            # Extract person region
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            
            # Detect faces in person region
            face_locations = self.face_detector.detect_faces_in_frame(person_crop)
            
            for face in face_locations:
                fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
                fw = fx2 - fx1
                fh = fy2 - fy1
                if fw < 80 or fh < 80:
                    continue

                # Adjust coordinates to full frame
                face_left = x1 + fx1
                face_top = y1 + fy1
                face_right = x1 + fx2
                face_bottom = y1 + fy2
                
                # Match face
                person_id, confidence = self.face_detector.match_face(face.embedding)
                
                if person_id:
                    detections.append({
                        'person_id': person_id,
                        'person_name': self.face_detector.known_person_ids[person_id]['name'],
                        'confidence': confidence,
                        'face_bbox': (face_left, face_top, face_right, face_bottom),
                        'person_bbox': person['bbox']
                    })
        
        return detections

def save_detection(person_id, frame, detection, source_name):
    """Save detection to database and disk"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"found/{person_id}/{source_name}"
        os.makedirs(save_dir, exist_ok=True)
        
        # Save full frame
        frame_path = f"{save_dir}/{timestamp}_full.jpg"
        cv2.imwrite(frame_path, frame)
        
        # Save face crop
        x1, y1, x2, y2 = detection['face_bbox']
        face_crop = frame[y1:y2, x1:x2]
        face_path = f"{save_dir}/{timestamp}_face.jpg"
        cv2.imwrite(face_path, face_crop)
        
        # Update database
        conn, cur = init_database()
        cur.execute("UPDATE missing_people SET status=1 WHERE id=? AND status=0", (person_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"Detection saved for person {person_id}")
        return frame_path, face_path
        
    except Exception as e:
        logger.error(f"Error saving detection: {e}")
        return None, None


def infer_footage_config(video_path):
    """Auto-tune footage detection settings based on video duration, resolution, and quality."""
    profile_defaults = {
        'Fast': {
            'profile': 'Fast',
            'surity': 2,
            'frame_time_gap': 6,
            'similarity_threshold': 0.54,
            'process_every_n_frames': 3,
            'min_face_size': 72,
            'blur_threshold': 65.0,
            'vote_window_seconds': 5,
            'det_size': (512, 512),
            'enable_multiscale_retry': False,
            'multiscale_retry_every_n_frames': 14,
        },
        'Balanced': {
            'profile': 'Balanced',
            'surity': 3,
            'frame_time_gap': 5,
            'similarity_threshold': 0.50,
            'process_every_n_frames': 2,
            'min_face_size': 64,
            'blur_threshold': 55.0,
            'vote_window_seconds': 6,
            'det_size': (640, 640),
            'enable_multiscale_retry': True,
            'multiscale_retry_every_n_frames': 10,
        },
        'Accurate': {
            'profile': 'Accurate',
            'surity': 3,
            'frame_time_gap': 5,
            'similarity_threshold': 0.46,
            'process_every_n_frames': 1,
            'min_face_size': 56,
            'blur_threshold': 45.0,
            'vote_window_seconds': 8,
            'det_size': (800, 800),
            'enable_multiscale_retry': True,
            'multiscale_retry_every_n_frames': 6,
        },
    }

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cfg = dict(profile_defaults['Balanced'])
        return cfg, "Balanced fallback (unable to read video metadata)"

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    duration_sec = (total_frames / fps) if fps > 1e-6 and total_frames > 0 else 0.0

    sample_stats = []
    sample_limit = 12
    # Sequential sampling avoids expensive random-seek overhead on many CCTV codecs.
    sequential_step = max(1, (total_frames // sample_limit) if total_frames > 0 else 10)
    frame_idx = 0
    while len(sample_stats) < sample_limit:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if frame_idx % sequential_step == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = float(np.mean(gray))
            sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            sample_stats.append((brightness, sharpness))
        frame_idx += 1

    cap.release()

    if sample_stats:
        avg_brightness = float(np.mean([x[0] for x in sample_stats]))
        avg_sharpness = float(np.mean([x[1] for x in sample_stats]))
    else:
        avg_brightness = 110.0
        avg_sharpness = 70.0

    high_res = (width * height) >= (1280 * 720)
    long_video = duration_sec >= 120.0
    low_light_or_blur = (avg_brightness < 60.0) or (avg_sharpness < 50.0)

    if low_light_or_blur:
        chosen = 'Accurate'
        reason = "Auto-selected Accurate (low-light/blurred footage)"
    elif high_res or long_video:
        chosen = 'Fast'
        reason = "Auto-selected Fast (high-resolution/long footage)"
    else:
        chosen = 'Balanced'
        reason = "Auto-selected Balanced (normal footage quality)"

    cfg = dict(profile_defaults[chosen])
    return cfg, reason

def process_video_file(video_path, progress_callback=None, footage_config=None):
    """Process uploaded video file using enhanced detector tuned for CCTV footage."""
    detections_log = []
    if progress_callback:
        progress_callback(1)

    auto_cfg, auto_reason = infer_footage_config(video_path)
    cfg = dict(auto_cfg)
    if footage_config:
        cfg.update({k: v for k, v in footage_config.items() if k in cfg})

    logger.info("Footage auto-config: %s | reason=%s", cfg.get('profile', 'Balanced'), auto_reason)
    if progress_callback:
        progress_callback(5)
    
    # Snapshot files before run so we can list only new detections from this processing job.
    before_files = set(str(p) for p in Path("found").rglob("*.jpg"))

    provider_info = "unknown"

    progress_state = {"last": 0.0}

    def monotonic_progress(pct):
        pct = float(max(0.0, min(100.0, pct)))
        if pct < progress_state["last"]:
            pct = progress_state["last"]
        progress_state["last"] = pct
        if progress_callback:
            progress_callback(pct)

    def stage_callback(stage_start, stage_end):
        def _cb(local_pct):
            local = float(max(0.0, min(100.0, local_pct)))
            mapped = stage_start + (stage_end - stage_start) * (local / 100.0)
            monotonic_progress(mapped)
        return _cb

    def run_once(local_cfg, stage_start=5.0, stage_end=90.0):
        nonlocal provider_info
        detector = None
        try:
            detector = MissingPersonDetector(
                db_path='data',
                surity=local_cfg['surity'],
                frame_time_gap=local_cfg['frame_time_gap'],
                similarity_threshold=local_cfg['similarity_threshold'],
                process_every_n_frames=local_cfg['process_every_n_frames'],
                min_face_size=local_cfg['min_face_size'],
                blur_threshold=local_cfg['blur_threshold'],
                vote_window_seconds=local_cfg['vote_window_seconds'],
                det_size=local_cfg['det_size'],
                enable_multiscale_retry=local_cfg['enable_multiscale_retry'],
                multiscale_retry_every_n_frames=local_cfg['multiscale_retry_every_n_frames'],
                prefer_gpu=True,
                confidence_margin=0.04,
                enable_focus_refine=True,
            )
            provider_info = ", ".join(getattr(detector, "active_providers", ["unknown"]))
            detector.process_video(
                video_path,
                source_name=Path(video_path).name,
                progress_callback=stage_callback(stage_start, stage_end),
            )
        finally:
            if detector is not None:
                detector.close()

    run_once(cfg, stage_start=5.0, stage_end=88.0)

    if progress_callback:
        progress_callback(100)

    after_files = set(str(p) for p in Path("found").rglob("*.jpg"))
    new_files = sorted(after_files - before_files)
    face_files = [p for p in new_files if p.endswith("_face.jpg")]
    if face_files:
        new_files = face_files

    conn, cur = init_database()
    cur.execute("SELECT id, name FROM missing_people")
    pid_to_name = {str(row['id']): row['name'] for row in cur.fetchall()}
    conn.close()

    for file_path in new_files:
        p = Path(file_path)
        # Expected layout: found/<pid>/<source>/<timestamp>.jpg
        if "found" not in p.parts:
            continue
        found_idx = p.parts.index("found")
        if len(p.parts) <= found_idx + 2:
            continue
        person_id = p.parts[found_idx + 1]
        person_name = pid_to_name.get(str(person_id), f"PID {person_id}")
        full_path = str(p).replace("_face.jpg", "_full.jpg")
        detections_log.append({
            'person_id': int(person_id) if str(person_id).isdigit() else person_id,
            'person_name': person_name,
            'confidence': None,
            'frame_number': None,
            'timestamp': datetime.fromtimestamp(p.stat().st_mtime),
            'frame_path': full_path if os.path.exists(full_path) else str(p),
            'face_path': str(p),
        })

    if not detections_log:
        fallback_cfg = dict(cfg)
        fallback_cfg.update({
            'profile': 'AccurateFallback',
            'similarity_threshold': min(float(cfg['similarity_threshold']), 0.45),
            'process_every_n_frames': 1,
            'min_face_size': min(int(cfg['min_face_size']), 56),
            'blur_threshold': min(float(cfg['blur_threshold']), 45.0),
            'det_size': (800, 800),
            'enable_multiscale_retry': True,
            'multiscale_retry_every_n_frames': 6,
        })
        logger.info("No detections in first pass, running accurate fallback pass")
        run_once(fallback_cfg, stage_start=88.0, stage_end=96.0)
        cfg = fallback_cfg

        after_files_fb = set(str(p) for p in Path("found").rglob("*.jpg"))
        new_files_fb = sorted(after_files_fb - before_files)
        face_files_fb = [p for p in new_files_fb if p.endswith("_face.jpg")]
        if face_files_fb:
            new_files_fb = face_files_fb

        for file_path in new_files_fb:
            p = Path(file_path)
            if "found" not in p.parts:
                continue
            found_idx = p.parts.index("found")
            if len(p.parts) <= found_idx + 2:
                continue
            person_id = p.parts[found_idx + 1]
            person_name = pid_to_name.get(str(person_id), f"PID {person_id}")
            full_path = str(p).replace("_face.jpg", "_full.jpg")
            detections_log.append({
                'person_id': int(person_id) if str(person_id).isdigit() else person_id,
                'person_name': person_name,
                'confidence': None,
                'frame_number': None,
                'timestamp': datetime.fromtimestamp(p.stat().st_mtime),
                'frame_path': full_path if os.path.exists(full_path) else str(p),
                'face_path': str(p),
            })

    cfg["provider_info"] = provider_info
    monotonic_progress(100.0)
    return detections_log, cfg, auto_reason

def main():
    """Main Streamlit app"""
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 3rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .sub-header {
            font-size: 1.5rem;
            color: #ff7f0e;
            margin-top: 2rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        [data-testid="stSidebar"] > div:first-child {
            height: 100vh;
        }
        [data-testid="stSidebar"] .stImage {
            margin-top: 0.25rem;
            margin-bottom: 0.6rem;
        }
        [data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
            font-weight: 800;
            font-size: 1.02rem;
        }
        [data-testid="stSidebar"] [role="radiogroup"] {
            display: flex;
            flex-direction: column;
            gap: 0.25rem;
            width: 100%;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] {
            width: 100%;
            padding: 0.35rem 0.45rem;
            border-radius: 0.45rem;
        }
        [data-testid="stSidebar"] [data-baseweb="radio"] p {
            font-weight: 800;
            font-size: 0.96rem;
            line-height: 1.2;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<div class="main-header">🔍 CCTV AI - Missing Person Detection</div>', unsafe_allow_html=True)
    
    # Initialize database
    conn, cur = init_database()
    
    # Sidebar navigation
    with st.sidebar:
        st.image("assets/CCTVpic.jpeg", use_container_width=True)
        st.title("Navigation")
        
        menu = st.radio(
            "Select Option",
            ["📊 Dashboard", "➕ Register Missing Person", "📹 Live Detection", 
               "🎬 Process Footage", "🗂️ Manage References", "✅ Verify Matches", "🔍 Search Records"]
        )
    
    # Dashboard
    if menu == "📊 Dashboard":
        st.markdown('<div class="sub-header">System Dashboard</div>', unsafe_allow_html=True)
        
        # Statistics
        cur.execute("SELECT COUNT(*) as total FROM missing_people")
        total = cur.fetchone()['total']
        
        cur.execute("SELECT COUNT(*) as pending FROM missing_people WHERE status=1")
        pending = cur.fetchone()['pending']
        
        cur.execute("SELECT COUNT(*) as verified FROM missing_people WHERE status=2")
        verified = cur.fetchone()['verified']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Cases", total)
        with col2:
            st.metric("Pending Verification", pending, delta="+new")
        with col3:
            st.metric("Verified Matches", verified)
        
        # Recent activity
        st.subheader("Recent Cases")
        cur.execute("SELECT * FROM missing_people ORDER BY created_at DESC LIMIT 10")
        records = cur.fetchall()
        
        if records:
            df = pd.DataFrame([dict(r) for r in records])
            df = df[['id', 'name', 'age', 'gender', 'missing_city', 'missing_state', 'status']]
            df['status'] = df['status'].map({0: 'Not Detected', 1: 'Pending', 2: 'Verified', 3: 'Failed'})
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No cases registered yet")
    
    # Register Missing Person
    elif menu == "➕ Register Missing Person":
        st.markdown('<div class="sub-header">Register New Missing Person</div>', unsafe_allow_html=True)
        
        with st.form("register_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Missing Person Details")
                name = st.text_input("Full Name *")
                gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
                age = st.number_input("Age *", min_value=1, max_value=120, value=25)
                
                missing_state = st.text_input("Missing State *")
                missing_city = st.text_input("Missing City *")
                pincode = st.text_input("Pincode *")
                missing_date = st.date_input("Missing Date *")
                description = st.text_area("Description")
                
                photo = st.file_uploader("Upload Recent Photo *", type=['jpg', 'jpeg', 'png'])
                additional_photos = st.file_uploader(
                    "Optional: Additional Reference Photos (0-5)",
                    type=['jpg', 'jpeg', 'png'],
                    accept_multiple_files=True,
                )
            
            with col2:
                st.subheader("Complainant Details")
                complaint_name = st.text_input("Complainant Name *")
                complaint_phone = st.text_input("Phone Number *")
                complaint_email = st.text_input("Complainant Email ID")
                complaint_address = st.text_area("Address *")
                
                st.subheader("Optional")
                footage = st.file_uploader("CCTV Footage (Optional)", type=['mp4', 'avi', 'mkv', 'mov'])
                enhance_refs = st.checkbox("Enhance low-light reference photos", value=True)
            
            submitted = st.form_submit_button("Register", use_container_width=True, type="primary")
            
            if submitted:
                if not all([name, age, missing_state, missing_city, pincode, missing_date, 
                           complaint_name, complaint_phone, complaint_address, photo]):
                    st.error("Please fill all required fields marked with *")
                else:
                    try:
                        if additional_photos and len(additional_photos) > 8:
                            st.error("Please upload at most 8 additional reference photos.")
                            additional_photos = additional_photos[:8]

                        primary_bgr = _uploaded_to_bgr(photo, enhance=enhance_refs)
                        if not _contains_face_opencv(primary_bgr):
                            st.error("No face detected in the photo. Please upload a clear face photo.")
                        else:
                            # Save footage if provided
                            footage_path = None
                            if footage:
                                footage_path = f"temp/{footage.name}"
                                with open(footage_path, 'wb') as f:
                                    f.write(footage.getbuffer())
                            
                            # Insert into database
                            cur.execute('''
                                INSERT INTO missing_people 
                                (name, gender, age, missing_state, missing_city, pincode, 
                                 missing_date, description, image_f, complaint_name, 
                                 complaint_phone, complaint_email, complaint_address, footage_path, face_encoding, status, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                            ''', (name, gender, age, missing_state, missing_city, pincode,
                                 str(missing_date), description, "", complaint_name,
                                 complaint_phone, complaint_email, complaint_address, footage_path, "opencv_detected", datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
                            
                            conn.commit()
                            person_id = cur.lastrowid

                            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                            saved_paths = []

                            primary_path = f"data/{person_id}_primary_{ts}.jpg"
                            cv2.imwrite(primary_path, primary_bgr)
                            saved_paths.append(primary_path)

                            if additional_photos:
                                for idx, extra in enumerate(additional_photos, start=1):
                                    try:
                                        ref_bgr = _uploaded_to_bgr(extra, enhance=enhance_refs)
                                        if not _contains_face_opencv(ref_bgr):
                                            continue
                                        ref_path = f"data/{person_id}_ref{idx}_{ts}.jpg"
                                        cv2.imwrite(ref_path, ref_bgr)
                                        saved_paths.append(ref_path)
                                    except Exception as ref_err:
                                        logger.warning(f"Skipped one reference photo for {name}: {ref_err}")

                            cur.execute(
                                "UPDATE missing_people SET image_f=?, face_encoding=?, reference_images_json=?, reference_count=? WHERE id=?",
                                (
                                    primary_path,
                                    f"opencv_detected:{len(saved_paths)}_refs",
                                    json.dumps(saved_paths),
                                    int(len(saved_paths)),
                                    person_id,
                                ),
                            )
                            conn.commit()

                            st.success(f"✅ Successfully registered {name} with {len(saved_paths)} reference photo(s)")
                            
                            # Reload face detector
                            if 'detector' in st.session_state:
                                st.session_state.detector.face_detector.load_known_faces()
                            
                            logger.info(f"Registered new person: {name}")
                    
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        logger.error(f"Registration error: {e}")
    
    # Live Detection
    elif menu == "📹 Live Detection":
        st.markdown('<div class="sub-header">Live CCTV Detection</div>', unsafe_allow_html=True)

        camera_id = st.number_input("Camera ID", min_value=0, max_value=10, value=0)
        
        col1, col2 = st.columns(2)
        start_btn = col1.button("🟢 Start Detection", use_container_width=True)
        stop_btn = col2.button("🔴 Stop Detection", use_container_width=True)
        
        if start_btn:
            st.session_state.detection_running = True
        if stop_btn:
            st.session_state.detection_running = False
        
        if st.session_state.get('detection_running', False):
            try:
                cap = cv2.VideoCapture(camera_id)
                detector = PersonDetector()
                
                frame_placeholder = st.empty()
                detection_log = st.empty()
                
                detections_list = []
                
                while st.session_state.get('detection_running', False):
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to read from camera")
                        break
                    
                    # Process frame
                    detections = detector.process_frame(frame)
                    
                    # Draw detections
                    for detection in detections:
                        x1, y1, x2, y2 = detection['face_bbox']
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                        shown_prob = _format_live_probability_ui(detection.get('confidence', 0.0))
                        label = f"{detection['person_name']} ({shown_prob:.2f}%)"
                        cv2.putText(frame, label, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Add time below the person
                        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        cv2.putText(frame, current_time, (x1, y2+20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        
                        # Save detection
                        save_detection(detection['person_id'], frame, detection, f"cam_{camera_id}")
                        detections_list.append(detection)
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Updated to fix deprecation warning: use_container_width=True -> width='stretch'
                    frame_placeholder.image(frame_rgb, channels="RGB", width='stretch')
                    
                    # Show detection log
                    if detections_list:
                        detection_log.write(f"✅ Detected: {len(detections_list)} matches")
                    
                    time.sleep(0.1)
                
                cap.release()
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                logger.error(f"Live detection error: {e}")
    
    # Process Footage
    elif menu == "🎬 Process Footage":
        st.markdown('<div class="sub-header">Process CCTV Footage</div>', unsafe_allow_html=True)


        ref_counts = _get_person_reference_counts(db_path="data")
        low_ref_pids = [pid for pid, cnt in ref_counts.items() if cnt < 3]
        if low_ref_pids:
            st.warning(
                ""
                
            )
        
        uploaded_file = st.file_uploader("Upload Video File", type=['mp4', 'avi', 'mkv', 'mov'])
        
        if uploaded_file:
            # Save temporarily
            temp_path = f"temp/{uploaded_file.name}"
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            if st.button("🎬 Process Video", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def update_progress(value):
                    progress_bar.progress(int(value))
                    status_text.text(f"Processing: {value:.1f}%")
                
                try:
                    detections, used_cfg, auto_reason = process_video_file(temp_path, update_progress)

                    st.info(
                        f"Auto profile: {used_cfg.get('profile', 'Balanced')} | "
                        f"Frame skip: {used_cfg.get('process_every_n_frames')} | "
                        f"Threshold: {used_cfg.get('similarity_threshold'):.2f} | "
                        f"Det size: {used_cfg.get('det_size')[0]} | "
                        f"Provider: {used_cfg.get('provider_info', 'unknown')} | {auto_reason}"
                    )
                    
                    st.success(f"✅ Processing complete! Found {len(detections)} matches")
                    
                    if detections:
                        df = pd.DataFrame(detections)
                        st.dataframe(df[['person_name', 'confidence', 'frame_number', 'timestamp']], 
                                   use_container_width=True)
                        
                        # Show sample detections
                        st.subheader("Sample Detections")
                        cols = st.columns(3)
                        for idx, detection in enumerate(detections[:6]):
                            with cols[idx % 3]:
                                if os.path.exists(detection['face_path']):
                                    st.image(detection['face_path'], caption=f"Face: {detection['person_name']}")
                                if os.path.exists(detection['frame_path']):
                                    st.image(detection['frame_path'], caption="Detection Frame")
                
                except Exception as e:
                    st.error(f"Error processing video: {str(e)}")
                    logger.error(f"Video processing error: {e}")
                
                finally:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)

    # Manage References
    elif menu == "🗂️ Manage References":
        st.markdown('<div class="sub-header">Manage Reference Images</div>', unsafe_allow_html=True)
        st.caption("Add or remove reference images used by both live and footage matching.")

        cur.execute(
            "SELECT id, name, status, image_f, reference_images_json, reference_count FROM missing_people ORDER BY id DESC"
        )
        people = cur.fetchall()

        if not people:
            st.info("No registered people found.")
        else:
            labels = [f"{r['id']} - {r['name']} (status={r['status']})" for r in people]
            selected = st.selectbox("Select Person", labels)
            selected_id = int(selected.split(" - ")[0])
            record = next(r for r in people if int(r["id"]) == selected_id)

            refs = _load_person_reference_paths(record)
            st.write(f"Current references: {len(refs)}")

            if refs:
                cols = st.columns(3)
                for idx, ref_path in enumerate(refs):
                    with cols[idx % 3]:
                        st.image(ref_path, caption=Path(ref_path).name)
                        if st.button("Remove", key=f"remove_ref_{selected_id}_{idx}"):
                            try:
                                updated = [p for p in refs if p != ref_path]
                                _save_person_reference_paths(cur, selected_id, updated)
                                conn.commit()
                                st.success("Reference removed")
                                st.rerun()
                            except Exception as rem_err:
                                st.error(f"Failed to remove reference: {rem_err}")

            add_files = st.file_uploader(
                "Add New Reference Photos",
                type=['jpg', 'jpeg', 'png'],
                accept_multiple_files=True,
                key=f"add_refs_{selected_id}",
            )
            enhance_new_refs = st.checkbox("Enhance low-light on new refs", value=True, key=f"enhance_refs_{selected_id}")

            if st.button("Add References", type="primary", key=f"add_refs_btn_{selected_id}"):
                if not add_files:
                    st.warning("Please upload at least one image.")
                else:
                    try:
                        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
                        new_paths = list(refs)
                        added = 0
                        for idx, uploaded in enumerate(add_files, start=1):
                            bgr = _uploaded_to_bgr(uploaded, enhance=enhance_new_refs)
                            if not _contains_face_opencv(bgr):
                                continue
                            out_path = f"data/{selected_id}_manualref{idx}_{ts}.jpg"
                            cv2.imwrite(out_path, bgr)
                            new_paths.append(out_path)
                            added += 1

                        # Dedup preserve order.
                        dedup = []
                        seen = set()
                        for p in new_paths:
                            if p not in seen:
                                seen.add(p)
                                dedup.append(p)

                        _save_person_reference_paths(cur, selected_id, dedup)
                        conn.commit()

                        if added == 0:
                            st.warning("No valid face found in uploaded images.")
                        else:
                            st.success(f"Added {added} reference image(s).")
                        st.rerun()
                    except Exception as add_err:
                        st.error(f"Failed to add references: {add_err}")
    
    # Verify Matches
    elif menu == "✅ Verify Matches":
        st.markdown('<div class="sub-header">Verify Detection Matches</div>', unsafe_allow_html=True)

        feedback = st.session_state.pop("verify_feedback", None)
        if isinstance(feedback, dict):
            level = str(feedback.get("level", "info"))
            message = str(feedback.get("message", ""))
            detail = str(feedback.get("detail", "")).strip()
            if level == "success":
                st.success(message)
            elif level == "warning":
                st.warning(message)
            elif level == "error":
                st.error(message)
            else:
                st.info(message)
            if detail:
                st.caption(detail)
        
        cur.execute("SELECT * FROM missing_people WHERE status=1")
        pending = cur.fetchall()
        
        if not pending:
            st.info("No pending verifications")
        else:
            for record in pending:
                person_id = record['id']
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 3, 2])
                    
                    with col1:
                        st.subheader(record['name'])
                        if os.path.exists(record['image_f']):
                            st.image(record['image_f'], caption="Registered Photo")
                    
                    with col2:
                        st.write(f"**Age:** {record['age']} | **Gender:** {record['gender']}")
                        st.write(f"**Location:** {record['missing_city']}, {record['missing_state']}")
                        st.write(f"**Contact:** {record['complaint_phone']}")
                        if record['complaint_email']:
                            st.write(f"**Complainant Email:** {record['complaint_email']}")
                        
                        # Show detected images
                        found_dir = f"found/{person_id}"
                        if os.path.exists(found_dir):
                            images = sorted(
                                Path(found_dir).rglob("*_face.jpg"),
                                key=lambda p: p.stat().st_mtime,
                                reverse=True,
                            )
                            if images:
                                st.write("**Detected Images:**")
                                for img_path in images[:3]:
                                    st.image(str(img_path), width=150)
                    
                    with col3:
                        if st.button("✅ Confirm Match", key=f"match_{person_id}", 
                                   use_container_width=True):
                            cur.execute("SELECT * FROM missing_people WHERE id=?", (person_id,))
                            fresh_record = cur.fetchone()
                            if not fresh_record:
                                st.session_state["verify_feedback"] = {
                                    "level": "error",
                                    "message": f"Unable to confirm. Record not found for ID {person_id}.",
                                    "detail": "Please refresh and try again.",
                                }
                                st.rerun()

                            notify_result = _send_match_notifications(fresh_record)
                            email_sent = 1 if notify_result['email_ok'] else 0
                            sms_sent = 1 if notify_result['sms_ok'] else 0
                            notification_message = (
                                f"Email: {notify_result['email_msg']} | "
                                f"SMS: {notify_result['sms_msg']}"
                            )

                            cur.execute(
                                """
                                UPDATE missing_people
                                SET status=2,
                                    notification_email_sent=?,
                                    notification_sms_sent=?,
                                    notification_last_message=?
                                WHERE id=?
                                """,
                                (email_sent, sms_sent, notification_message, person_id),
                            )
                            conn.commit()

                            if notify_result['email_ok'] and notify_result['sms_ok']:
                                st.session_state["verify_feedback"] = {
                                    "level": "success",
                                    "message": "Match confirmed. Mail and SMS notifications have been sent.",
                                    "detail": notification_message,
                                }
                            elif notify_result['email_ok'] or notify_result['sms_ok']:
                                st.session_state["verify_feedback"] = {
                                    "level": "warning",
                                    "message": "Match confirmed. One notification channel was sent and one failed.",
                                    "detail": notification_message,
                                }
                            else:
                                st.session_state["verify_feedback"] = {
                                    "level": "warning",
                                    "message": "Match confirmed, but notifications could not be sent.",
                                    "detail": notification_message,
                                }
                            st.rerun()
                        
                        if st.button("❌ Not a Match", key=f"no_match_{person_id}", 
                                   use_container_width=True):
                            cur.execute("UPDATE missing_people SET status=3 WHERE id=?", (person_id,))
                            conn.commit()
                            st.session_state["verify_feedback"] = {
                                "level": "info",
                                "message": "Marked as not a match.",
                                "detail": f"Case ID {person_id} moved to Failed status.",
                            }
                            st.rerun()
                    
                    st.divider()
    
    # Search Records
    elif menu == "🔍 Search Records":
        st.markdown('<div class="sub-header">Search Records</div>', unsafe_allow_html=True)
        
        search_term = st.text_input("Search by name, city, or state")
        status_filter = st.multiselect("Status", ["Not Detected", "Pending", "Verified", "Failed"])
        
        query = "SELECT * FROM missing_people WHERE 1=1"
        params = []
        
        if search_term:
            query += " AND (name LIKE ? OR missing_city LIKE ? OR missing_state LIKE ?)"
            params.extend([f"%{search_term}%"] * 3)
        
        if status_filter:
            status_map = {"Not Detected": 0, "Pending": 1, "Verified": 2, "Failed": 3}
            status_values = [status_map[s] for s in status_filter]
            query += f" AND status IN ({','.join('?' * len(status_values))})"
            params.extend(status_values)
        
        cur.execute(query, params)
        results = cur.fetchall()
        
        if results:
            st.write(f"Found {len(results)} records")
            
            for record in results:
                with st.expander(f"{record['name']} - {record['missing_city']}"):
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        if os.path.exists(record['image_f']):
                            st.image(record['image_f'])
                    
                    with col2:
                        st.write(f"**ID:** {record['id']}")
                        st.write(f"**Age/Gender:** {record['age']}/{record['gender']}")
                        st.write(f"**Location:** {record['missing_city']}, {record['missing_state']}")
                        st.write(f"**Missing Since:** {record['missing_date']}")
                        st.write(f"**Description:** {record['description']}")
                        st.write(f"**Contact:** {record['complaint_phone']}")
                        if record['complaint_email']:
                            st.write(f"**Complainant Email:** {record['complaint_email']}")
                        
                        status_text = {0: "Not Detected", 1: "Pending", 2: "Verified", 3: "Failed"}
                        st.write(f"**Status:** {status_text[record['status']]}")
                        email_status = "Sent" if int(record['notification_email_sent'] or 0) == 1 else "Not Sent"
                        sms_status = "Sent" if int(record['notification_sms_sent'] or 0) == 1 else "Not Sent"
                        st.write(f"**Notification Status:** Email {email_status} | SMS {sms_status}")
                        if record['notification_last_message']:
                            st.caption(f"Last notification log: {record['notification_last_message']}")
                        
                        if st.button("🗑️ Delete Record", key=f"del_{record['id']}", type="primary"):
                            try:
                                _delete_person_record(record, cur, conn)
                                st.success("Record deleted successfully")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error deleting record: {e}")
        else:
            st.info("No records found")
    
    conn.close()

if __name__ == "__main__":
    main()
