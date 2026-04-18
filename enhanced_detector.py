"""Unified InsightFace detector for both live camera and CCTV footage."""

from argparse import ArgumentParser
from collections import defaultdict, deque
from datetime import datetime
from json import dumps, loads
from os import makedirs, remove
from os.path import basename, exists, isdir, isfile
from shutil import rmtree
from threading import Event, Thread
from traceback import format_exc
import glob
import logging
import sqlite3
import itertools

import cv2
import numpy as np
from insightface.app import FaceAnalysis

try:
    import onnxruntime as ort
except Exception:  # pragma: no cover
    ort = None

try:
    from scipy.optimize import linear_sum_assignment
except Exception:  # pragma: no cover
    linear_sum_assignment = None

try:
    from filterpy.kalman import KalmanFilter
except Exception:  # pragma: no cover
    KalmanFilter = None


found_dir = "found"
makedirs("./logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("./logs/detection.log"), logging.StreamHandler()],
)


def _safe_norm(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def cosine_similarity(a, b):
    a_n = _safe_norm(a)
    b_n = _safe_norm(b)
    return float(np.dot(a_n, b_n))


def _iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


class _FootageTrack:
    def __init__(self, track_id, bbox, frame_index):
        self.track_id = int(track_id)
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.hits = 1
        self.missed = 0
        self.last_seen = int(frame_index)
        self.last_embedding = None
        self.pid = None
        self.score = -1.0
        self.last_reid_frame = -1
        self.kf = self._create_kf(self.bbox)

    @staticmethod
    def _create_kf(bbox):
        if KalmanFilter is None:
            return None
        x1, y1, x2, y2 = [float(v) for v in bbox]
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)

        kf = KalmanFilter(dim_x=8, dim_z=4)
        kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 0],
                [0, 0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 0, 1],
            ],
            dtype=np.float32,
        )
        kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0],
            ],
            dtype=np.float32,
        )
        kf.P *= 10.0
        kf.Q *= 0.01
        kf.R *= 0.1
        kf.x = np.array([[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32)
        return kf

    def predict(self):
        if self.kf is None:
            return
        self.kf.predict()
        cx, cy, w, h = [float(v) for v in self.kf.x[:4, 0]]
        x1 = cx - w * 0.5
        y1 = cy - h * 0.5
        x2 = cx + w * 0.5
        y2 = cy + h * 0.5
        self.bbox = np.array([x1, y1, x2, y2], dtype=np.float32)

    def update(self, bbox, frame_index, embedding):
        self.bbox = np.asarray(bbox, dtype=np.float32)
        self.hits += 1
        self.missed = 0
        self.last_seen = int(frame_index)
        self.last_embedding = embedding
        if self.kf is not None:
            x1, y1, x2, y2 = [float(v) for v in self.bbox]
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            w = max(1.0, x2 - x1)
            h = max(1.0, y2 - y1)
            self.kf.update(np.array([[cx], [cy], [w], [h]], dtype=np.float32))


class _FootageTracker:
    def __init__(self, iou_threshold=0.25, max_missed=12, min_hits=2):
        self.iou_threshold = float(iou_threshold)
        self.max_missed = int(max_missed)
        self.min_hits = int(min_hits)
        self._id_gen = itertools.count(1)
        self.tracks = []

    def _associate(self, det_boxes):
        if not self.tracks or not det_boxes:
            return [], list(range(len(self.tracks))), list(range(len(det_boxes)))

        cost = np.ones((len(self.tracks), len(det_boxes)), dtype=np.float32)
        for ti, trk in enumerate(self.tracks):
            for di, dbox in enumerate(det_boxes):
                cost[ti, di] = 1.0 - _iou(trk.bbox, dbox)

        if linear_sum_assignment is not None:
            row_ind, col_ind = linear_sum_assignment(cost)
            pairs = list(zip(row_ind.tolist(), col_ind.tolist()))
        else:
            pairs = []
            used_t = set()
            used_d = set()
            flat = [
                (cost[t, d], t, d)
                for t in range(cost.shape[0])
                for d in range(cost.shape[1])
            ]
            flat.sort(key=lambda x: x[0])
            for _, t, d in flat:
                if t in used_t or d in used_d:
                    continue
                pairs.append((t, d))
                used_t.add(t)
                used_d.add(d)

        matches = []
        used_tracks = set()
        used_dets = set()
        for t, d in pairs:
            iou = 1.0 - float(cost[t, d])
            if iou < self.iou_threshold:
                continue
            matches.append((t, d))
            used_tracks.add(t)
            used_dets.add(d)

        unmatched_tracks = [i for i in range(len(self.tracks)) if i not in used_tracks]
        unmatched_dets = [i for i in range(len(det_boxes)) if i not in used_dets]
        return matches, unmatched_tracks, unmatched_dets

    def step(self, detections, frame_index):
        for trk in self.tracks:
            trk.predict()

        det_boxes = [d["bbox"] for d in detections]
        matches, unmatched_tracks, unmatched_dets = self._associate(det_boxes)

        matched_track_ids = set()
        for t_idx, d_idx in matches:
            det = detections[d_idx]
            self.tracks[t_idx].update(det["bbox"], frame_index, det["embedding"])
            matched_track_ids.add(self.tracks[t_idx].track_id)

        for t_idx in unmatched_tracks:
            self.tracks[t_idx].missed += 1

        for d_idx in unmatched_dets:
            det = detections[d_idx]
            trk = _FootageTrack(next(self._id_gen), det["bbox"], frame_index)
            trk.last_embedding = det["embedding"]
            self.tracks.append(trk)

        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        active_tracks = [
            t for t in self.tracks
            if t.hits >= self.min_hits and t.track_id in matched_track_ids and t.last_embedding is not None
        ]
        return active_tracks


class MissingPersonDetector:
    """High-accuracy face detector/recognizer using InsightFace embeddings."""

    def __init__(
        self,
        db_path="data",
        model_name="arcface",  # kept for backward compatibility
        detector_backend="retinaface",  # kept for backward compatibility
        surity=3,
        frame_time_gap=10,
        similarity_threshold=0.45,
        process_every_n_frames=2,
        min_face_size=80,
        blur_threshold=80.0,
        vote_window_seconds=5,
        det_size=(640, 640),
        enable_multiscale_retry=False,
        multiscale_retry_every_n_frames=8,
        prefer_gpu=True,
        confidence_margin=0.04,
        enable_focus_refine=True,
        focus_window=12,
        focus_step=2,
    ):
        self.db_path = db_path
        self.model_name = model_name
        self.detector_backend = detector_backend
        self.surity = max(1, int(surity))
        self.frame_time_gap = max(1, int(frame_time_gap))
        self.similarity_threshold = float(similarity_threshold)
        self.process_every_n_frames = max(1, int(process_every_n_frames))
        self.min_face_size = max(32, int(min_face_size))
        self.blur_threshold = float(blur_threshold)
        self.vote_window_seconds = max(1, int(vote_window_seconds))
        self.det_size = det_size
        self.enable_multiscale_retry = bool(enable_multiscale_retry)
        self.multiscale_retry_every_n_frames = max(1, int(multiscale_retry_every_n_frames))
        self.prefer_gpu = bool(prefer_gpu)
        self.confidence_margin = float(confidence_margin)
        self.enable_focus_refine = bool(enable_focus_refine)
        self.focus_window = max(2, int(focus_window))
        self.focus_step = max(1, int(focus_step))
        self._focus_candidate_frames = set()
        self.stop_event = Event()

        self.db_conn = sqlite3.connect("./Database/data.db", check_same_thread=False)
        self.db_conn.row_factory = sqlite3.Row
        self.cur = self.db_conn.cursor()

        makedirs(found_dir, exist_ok=True)

        providers = ["CPUExecutionProvider"]
        ctx_id = -1
        if self.prefer_gpu and ort is not None:
            try:
                available = set(ort.get_available_providers())
                if "CUDAExecutionProvider" in available:
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                    ctx_id = 0
            except Exception:
                providers = ["CPUExecutionProvider"]
                ctx_id = -1

        self.face_app = FaceAnalysis(name="buffalo_l", providers=providers)
        self.face_app.prepare(ctx_id=ctx_id, det_size=self.det_size)
        self.active_providers = providers
        logging.info("FaceAnalysis providers=%s ctx_id=%d", providers, ctx_id)

        self.gallery = defaultdict(list)
        self.rebuild_gallery()

        self.vote_state = defaultdict(deque)
        self.last_saved_at = {}

    @staticmethod
    def _apply_orientation(frame, orientation_mode):
        if orientation_mode == "normal":
            return frame
        if orientation_mode == "rot90cw":
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if orientation_mode == "rot180":
            return cv2.rotate(frame, cv2.ROTATE_180)
        if orientation_mode == "rot90ccw":
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orientation_mode == "flip_h":
            return cv2.flip(frame, 1)
        return frame

    def _select_best_orientation(self, cap, sample_count=10):
        """Probe a few frames to auto-select orientation for uploaded CCTV footage."""
        candidates = ["normal", "rot90cw", "rot180", "rot90ccw", "flip_h"]
        scores = {c: 0.0 for c in candidates}

        sampled = []
        for _ in range(max(1, sample_count)):
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            sampled.append(frame)

        if not sampled:
            return "normal"

        for frame in sampled:
            for mode in candidates:
                probe = self._apply_orientation(frame, mode)
                faces = self.face_app.get(probe)
                if not faces:
                    continue
                det_boost = max(float(getattr(f, "det_score", 0.0)) for f in faces)
                scores[mode] += float(len(faces)) + 0.25 * det_boost

        best_mode = max(scores, key=scores.get)
        logging.info("Auto-orientation selected: %s (scores=%s)", best_mode, scores)
        return best_mode

    def rebuild_gallery(self):
        """Build embedding gallery from registered person images."""
        self.gallery.clear()
        records = self.cur.execute(
            "SELECT id, image_f, reference_images_json FROM missing_people WHERE status IN (0,1)"
        ).fetchall()

        for record in records:
            pid = str(record["id"])
            candidates = glob.glob(f"{self.db_path}/{pid}_*.jpg")

            # Include schema-backed reference image paths.
            ref_json = record["reference_images_json"]
            if ref_json:
                try:
                    ref_paths = loads(ref_json)
                    if isinstance(ref_paths, list):
                        candidates.extend([p for p in ref_paths if isinstance(p, str)])
                except Exception as parse_err:
                    logging.warning("Bad reference_images_json for pid %s: %s", pid, parse_err)

            if record["image_f"] and exists(record["image_f"]):
                candidates.append(record["image_f"])

            # Deduplicate while preserving order.
            dedup_candidates = []
            seen = set()
            for c in candidates:
                if c not in seen:
                    seen.add(c)
                    dedup_candidates.append(c)
            candidates = dedup_candidates

            for img_path in candidates:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                faces = self.face_app.get(img)
                if not faces:
                    continue
                best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                self.gallery[pid].append(_safe_norm(best.embedding.astype(np.float32)))

        total = sum(len(v) for v in self.gallery.values())
        logging.info("Gallery rebuilt: %d people, %d embeddings", len(self.gallery), total)

    def _face_quality_ok(self, frame, bbox):
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if w < self.min_face_size or h < self.min_face_size:
            return False

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_score < self.blur_threshold:
            return False

        mean_brightness = float(gray.mean())
        if mean_brightness < 35 or mean_brightness > 220:
            return False

        return True

    def _best_match(self, embedding):
        emb = _safe_norm(embedding.astype(np.float32))
        best_pid, best_score = None, -1.0
        second_best = -1.0

        for pid, emb_list in self.gallery.items():
            if not emb_list:
                continue
            score = max(cosine_similarity(emb, ref_emb) for ref_emb in emb_list)
            if score > best_score:
                second_best = best_score
                best_score = score
                best_pid = pid
            elif score > second_best:
                second_best = score

        margin = best_score - max(-1.0, second_best)

        if best_pid is None or best_score < self.similarity_threshold:
            return None, best_score, margin
        if margin < self.confidence_margin:
            return None, best_score, margin
        return best_pid, best_score, margin

    def _detect_faces_with_optional_retry(self, frame, frame_index=0):
        """Run base face detection and optionally retry on an upscaled frame for hard CCTV angles."""
        faces = self.face_app.get(frame)
        if faces or not self.enable_multiscale_retry:
            return faces, 1.0

        # Throttle the expensive retry path to keep footage processing fast.
        if frame_index % self.multiscale_retry_every_n_frames != 0:
            return [], 1.0

        upscaled = cv2.resize(frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        faces_up = self.face_app.get(upscaled)
        if not faces_up:
            return [], 1.0

        return faces_up, (1.0 / 1.5)

    def sync_with_db(self, scan_info):
        try:
            active_pids = {
                str(row["id"])
                for row in self.cur.execute(
                    "SELECT id FROM missing_people WHERE status IN (0, 1)"
                ).fetchall()
            }

            removed_any = False
            for pid in list(scan_info.get("for_verification_pids", {}).keys()):
                if pid not in active_pids:
                    scan_info["for_verification_pids"].pop(pid, None)
                    removed_any = True

            fetch = self.cur.execute("SELECT * FROM missing_people WHERE status > 1").fetchall()
            for record in fetch:
                pid = str(record["id"])
                if pid in scan_info.get("for_verification_pids", {}):
                    scan_info["for_verification_pids"].pop(pid, None)
                    removed_any = True

            if removed_any:
                with open("scan_info.json", "w") as f:
                    f.write(dumps(scan_info))
        except Exception as exc:
            logging.error("Database sync error: %s", exc)

        return scan_info

    def _register_detection(self, frame, pid, bbox, score, scan_info, src_prefix):
        now = datetime.now()
        pid_votes = self.vote_state[pid]
        pid_votes.append(now)
        while pid_votes and (now - pid_votes[0]).total_seconds() > self.vote_window_seconds:
            pid_votes.popleft()

        if len(pid_votes) < self.surity:
            return scan_info

        last_saved = self.last_saved_at.get(pid)
        if last_saved and (now - last_saved).total_seconds() < self.frame_time_gap:
            return scan_info

        self.last_saved_at[pid] = now
        dt = now.strftime("%Y%m%dT%H%M%S")
        save_dir = f"{found_dir}/{pid}/{src_prefix}"
        makedirs(save_dir, exist_ok=True)

        x1, y1, x2, y2 = [int(v) for v in bbox]
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(frame.shape[1], x2 + pad)
        y2 = min(frame.shape[0], y2 + pad)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"PID:{pid} score:{score:.3f}",
            (x1, max(15, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

        annotated_full_path = f"{save_dir}/{dt}_full.jpg"
        face_crop_path = f"{save_dir}/{dt}_face.jpg"
        cv2.imwrite(annotated_full_path, frame)
        cv2.imwrite(face_crop_path, frame[y1:y2, x1:x2])

        if pid not in scan_info["for_verification_pids"]:
            scan_info["for_verification_pids"][pid] = []
        scan_info["for_verification_pids"][pid].append(face_crop_path)

        logging.info("Detected PID %s at score %.3f", pid, score)
        return scan_info

    def _update_db_pending_state(self, scan_info):
        pending_ids = [int(pid) for pid in scan_info.get("for_verification_pids", {}).keys() if str(pid).isdigit()]
        if not pending_ids:
            return
        placeholders = ",".join("?" for _ in pending_ids)
        sql = f"UPDATE missing_people SET status=1 WHERE id IN ({placeholders}) AND status=0"
        self.cur.execute(sql, pending_ids)
        self.db_conn.commit()

    def _build_focus_targets(self, candidate_frames, total_frames):
        targets = set()
        if total_frames <= 0:
            return []
        for center in candidate_frames:
            left = max(0, center - self.focus_window)
            right = min(total_frames - 1, center + self.focus_window)
            for idx in range(left, right + 1, self.focus_step):
                targets.add(idx)
        return sorted(targets)

    def _run_focus_refine(self, video_source, source_name, orientation_mode, scan_info, progress_callback=None):
        if not self._focus_candidate_frames:
            return scan_info

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            return scan_info

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        target_frames = self._build_focus_targets(self._focus_candidate_frames, total_frames)
        if not target_frames:
            cap.release()
            return scan_info

        src_prefix = basename(source_name).split(".")[0] if source_name else "cam0"
        base_thresh = self.similarity_threshold
        self.similarity_threshold = max(0.40, base_thresh - 0.03)
        seen = 0

        try:
            for idx in target_frames:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                if orientation_mode != "normal":
                    frame = self._apply_orientation(frame, orientation_mode)

                faces, scale_back = self._detect_faces_with_optional_retry(frame, idx)
                if not faces:
                    seen += 1
                    continue

                for face in faces:
                    bbox = np.array(face.bbox, dtype=np.float32)
                    if scale_back != 1.0:
                        bbox = bbox * scale_back
                    if not self._face_quality_ok(frame, bbox):
                        continue

                    pid, score, _margin = self._best_match(face.embedding)
                    if pid is None:
                        continue
                    active = self.cur.execute(
                        "SELECT id FROM missing_people WHERE id=? AND status IN (0,1)", (pid,)
                    ).fetchone()
                    if not active:
                        continue
                    scan_info = self._register_detection(frame, pid, bbox, score, scan_info, src_prefix)

                seen += 1
                if progress_callback and len(target_frames) > 0 and seen % 20 == 0:
                    # Focus pass contributes a smaller progress slice controlled by caller.
                    progress_callback(min(99.0, (seen / len(target_frames)) * 100.0))
        finally:
            self.similarity_threshold = base_thresh
            cap.release()

        return scan_info

    def process_video(self, video_source, source_name=None, progress_callback=None):
        if source_name is None:
            source_name = f"cam_{video_source}" if isinstance(video_source, int) else basename(str(video_source))
        src_prefix = basename(source_name).split(".")[0] if source_name else "cam0"
        is_live_source = isinstance(video_source, int)

        try:
            scan_info = loads(open("scan_info.json").read())
        except Exception:
            scan_info = {"for_verification_pids": {}, "error": ""}

        # For file-based footage runs, start with a clean pending map to avoid stale IDs
        # from previous sessions being marked as newly detected.
        if not is_live_source:
            scan_info["for_verification_pids"] = {}
            self.vote_state.clear()
            self.last_saved_at.clear()

        self.rebuild_gallery()
        scan_info = self.sync_with_db(scan_info)
        scan_info["error"] = ""

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            msg = f"Failed to open video source: {video_source}"
            logging.error(msg)
            scan_info["error"] = msg
            with open("scan_info.json", "w") as f:
                f.write(dumps(scan_info))
            return

        orientation_mode = "normal"
        if not is_live_source:
            orientation_mode = self._select_best_orientation(cap, sample_count=10)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        total_frames = 0
        if not is_live_source:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        frame_count = 0
        footage_tracker = _FootageTracker() if not is_live_source else None
        footage_reid_interval = 5
        self._focus_candidate_frames = set()
        logging.info("Starting detection on %s", source_name)

        try:
            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if isinstance(video_source, int):
                        continue
                    break

                frame_count += 1
                if orientation_mode != "normal":
                    frame = self._apply_orientation(frame, orientation_mode)
                if frame_count % self.process_every_n_frames != 0:
                    if progress_callback and total_frames > 0 and frame_count % 10 == 0:
                        progress_callback(min(99.0, (frame_count / total_frames) * 100.0))
                    continue

                faces, scale_back = self._detect_faces_with_optional_retry(frame, frame_count)
                if not faces:
                    if progress_callback and total_frames > 0 and frame_count % 10 == 0:
                        progress_callback(min(99.0, (frame_count / total_frames) * 100.0))
                    continue

                detections = []

                for face in faces:
                    bbox = np.array(face.bbox, dtype=np.float32)
                    if scale_back != 1.0:
                        bbox = bbox * scale_back
                    if not self._face_quality_ok(frame, bbox):
                        continue

                    detections.append({
                        "bbox": bbox,
                        "embedding": face.embedding,
                    })

                if not detections:
                    if progress_callback and total_frames > 0 and frame_count % 10 == 0:
                        progress_callback(min(99.0, (frame_count / total_frames) * 100.0))
                    continue

                if footage_tracker is not None:
                    active_tracks = footage_tracker.step(detections, frame_count)
                    for trk in active_tracks:
                        if trk.last_reid_frame < 0 or (frame_count - trk.last_reid_frame) >= footage_reid_interval:
                            pid, score, _margin = self._best_match(trk.last_embedding)
                            trk.last_reid_frame = frame_count
                            if pid is not None:
                                trk.pid = pid
                                trk.score = score
                            elif not is_live_source:
                                self._focus_candidate_frames.add(frame_count)

                        if trk.pid is None:
                            continue

                        active = self.cur.execute(
                            "SELECT id FROM missing_people WHERE id=? AND status IN (0,1)", (trk.pid,)
                        ).fetchone()
                        if not active:
                            continue

                        scan_info = self._register_detection(frame, trk.pid, trk.bbox, trk.score, scan_info, src_prefix)
                else:
                    for det in detections:
                        pid, score, _margin = self._best_match(det["embedding"])
                        if pid is None:
                            if not is_live_source:
                                self._focus_candidate_frames.add(frame_count)
                            continue

                        active = self.cur.execute(
                            "SELECT id FROM missing_people WHERE id=? AND status IN (0,1)", (pid,)
                        ).fetchone()
                        if not active:
                            continue

                        scan_info = self._register_detection(frame, pid, det["bbox"], score, scan_info, src_prefix)

                self._update_db_pending_state(scan_info)
                with open("scan_info.json", "w") as f:
                    f.write(dumps(scan_info))

                if progress_callback and total_frames > 0 and frame_count % 10 == 0:
                    progress_callback(min(99.0, (frame_count / total_frames) * 100.0))

            if not is_live_source and self.enable_focus_refine and len(self._focus_candidate_frames) > 0:
                logging.info("Running focused refine pass on %d candidate frames", len(self._focus_candidate_frames))
                scan_info = self._run_focus_refine(
                    video_source=video_source,
                    source_name=source_name,
                    orientation_mode=orientation_mode,
                    scan_info=scan_info,
                    progress_callback=progress_callback,
                )
                self._update_db_pending_state(scan_info)
                with open("scan_info.json", "w") as f:
                    f.write(dumps(scan_info))

        except Exception:
            scan_info["error"] = format_exc()
            logging.error("Error processing %s: %s", source_name, scan_info["error"])
        finally:
            cap.release()
            scan_info = self.sync_with_db(scan_info)
            with open("scan_info.json", "w") as f:
                f.write(dumps(scan_info))
            if progress_callback and total_frames > 0:
                progress_callback(100.0)
            logging.info("Stopped detection on %s", source_name)

    def start_live_detection(self, camera_id=0):
        thread = Thread(target=self.process_video, args=(camera_id, f"camera_{camera_id}"))
        thread.daemon = True
        thread.start()
        return thread

    def stop_detection(self):
        self.stop_event.set()
        logging.info("Detection stop requested")

    def close(self):
        self.stop_detection()
        if hasattr(self, "db_conn"):
            self.db_conn.close()


def main():
    parser = ArgumentParser(description="Missing Person Detection System")
    parser.add_argument("--cam_id", "-cid", type=int, default=None, help="Camera ID for live detection")
    parser.add_argument("--footage_path", "-fp", type=str, default=None, help="Path to video footage")
    parser.add_argument("--db_path", "-dp", type=str, default="data", help="Path to face database")
    parser.add_argument("--surity", "-s", type=int, default=3, choices=[1, 2, 3, 4, 5], help="Votes required")
    parser.add_argument("--frame_time_gap", "-ftg", type=int, default=10, help="Min seconds between saves")
    parser.add_argument("--threshold", "-t", type=float, default=0.45, help="Cosine threshold")
    parser.add_argument("--process_every_n_frames", "-pef", type=int, default=2, help="Frame skip value")
    parser.add_argument("--min_face_size", "-mfs", type=int, default=80, help="Min face size in pixels")
    parser.add_argument("--blur_threshold", "-bt", type=float, default=80.0, help="Laplacian blur threshold")
    args = parser.parse_args()

    if args.cam_id is None and args.footage_path is None:
        print("Error: Must specify either --cam_id or --footage_path")
        return

    detector = MissingPersonDetector(
        db_path=args.db_path,
        surity=args.surity,
        frame_time_gap=args.frame_time_gap,
        similarity_threshold=args.threshold,
        process_every_n_frames=args.process_every_n_frames,
        min_face_size=args.min_face_size,
        blur_threshold=args.blur_threshold,
    )

    try:
        if args.footage_path:
            detector.process_video(args.footage_path)
        else:
            detector.process_video(args.cam_id)
    except KeyboardInterrupt:
        logging.info("Detection interrupted by user")
    finally:
        detector.close()


if __name__ == "__main__":
    main()
