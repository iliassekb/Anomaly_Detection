#!/usr/bin/env python3
"""
Real-time Anomaly Detection — DefectVision Pro
Optimised for maximum FPS:
  • Dedicated capture thread  → inference never waits for I/O
  • MJPG codec + buffer=1    → always the freshest frame
  • Model warmup             → no cold-start penalty
  • FP16 on GPU (half=True)  → ~2× faster inference
  • deque for FPS            → O(1) rolling average
  • Runtime imgsz / skip     → interactive speed/quality trade-off
"""

import os
import re
import socket
import sys
import time
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


# ── Network auto-discovery ────────────────────────────────────────────────────

def _check_host(ip: str, port: int, timeout: float) -> str | None:
    """Return ip if port is open, else None."""
    try:
        with socket.create_connection((ip, port), timeout=timeout):
            return ip
    except OSError:
        return None


def discover_ip_webcam(port: int = 8080, timeout: float = 0.5) -> str | None:
    """
    Scan the local /24 subnet for a device listening on *port*.
    Returns the first HTTP-MJPEG URL found, or None.
    Typical scan time: 3-6 s for a /24 subnet.
    """
    # Determine local IP → derive subnet
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip: str = s.getsockname()[0]
        s.close()
    except OSError:
        print("  Could not determine local IP — skipping auto-discovery.")
        return None

    prefix = ".".join(local_ip.split(".")[:3])   # e.g. "192.168.1"
    candidates = [f"{prefix}.{i}" for i in range(1, 255)]

    print(f"  Scanning {prefix}.1-254 on port {port} …")
    found: list[str] = []
    with ThreadPoolExecutor(max_workers=80) as pool:
        futures = {pool.submit(_check_host, ip, port, timeout): ip
                   for ip in candidates}
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                found.append(result)

    if not found:
        return None

    # Prefer the IP that actually serves an IP-Webcam video feed
    for ip in found:
        url = f"http://{ip}:{port}/video"
        try:
            cap = cv2.VideoCapture(url, cv2.CAP_ANY)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            ok, _ = cap.read()
            cap.release()
            if ok:
                print(f"  ✓ IP Webcam found at {ip}:{port}")
                return url
        except Exception:
            pass

    # Fall back to the first open port (might be a different app)
    ip = found[0]
    print(f"  Found open port at {ip}:{port} (stream not verified)")
    return f"http://{ip}:{port}/video"


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "normal":  (0, 220, 80),
    "anomaly": (0, 60, 220),
    "text":    (255, 255, 255),
    "pass":    (0, 200, 80),
    "fail":    (0, 60, 220),
}

CLASS_COLORS = [
    (0,  60, 220), (220,  60,  0), (0, 180, 220), (180, 0, 220),
    (220, 180, 0), (0, 220, 160), (220, 80,  80), (80, 80, 220),
]


# ── Threaded frame grabber ────────────────────────────────────────────────────

class FrameGrabber(threading.Thread):
    """
    Runs camera capture on a background daemon thread.
    Accepts either a local camera index (int) or a network URL (str):
      • int  → USB/built-in webcam (DirectShow on Windows)
      • str  → RTSP / HTTP-MJPEG stream from phone or IP camera
    The main thread always reads the *latest* frame without blocking.
    Network drops are handled with auto-reconnect.
    """

    def __init__(self, source: int | str, width: int = 0, height: int = 0):
        super().__init__(daemon=True)
        self._source   = source
        self._width    = width
        self._height   = height
        self._is_url   = isinstance(source, str)
        self._cap      = self._open()

        self._frame: np.ndarray | None = None
        self._lock  = threading.Lock()
        self._stop  = threading.Event()

    # ── URL candidates to try in order ──────────────────────────────────────
    @staticmethod
    def _url_candidates(url: str) -> list[str]:
        """
        For IP Webcam (Android, port 8080) auto-generate fallback URLs.
        Other apps return only the original URL.
        """
        import re
        m = re.search(r"(\d+\.\d+\.\d+\.\d+):(\d+)", url)
        if m and m.group(2) == "8080":
            ip = m.group(1)
            return [
                f"rtsp://{ip}:8080/h264_ulaw.sdp",   # IP Webcam RTSP (H.264)
                f"http://{ip}:8080/video",             # IP Webcam HTTP-MJPEG
                f"http://{ip}:8080/videofeed",         # alternate MJPEG path
            ]
        return [url]

    def _try_open(self, url: str) -> cv2.VideoCapture | None:
        """Open one URL; return cap on success, None on failure (fast timeout)."""
        import os
        is_rtsp = url.startswith("rtsp")
        if is_rtsp:
            # TCP transport + 8-second connection timeout (µs)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = \
                "rtsp_transport;tcp|stimeout;8000000"
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(url, cv2.CAP_ANY)

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if cap.isOpened():
            ok, _ = cap.read()   # confirm we actually get frames
            if ok:
                return cap
            cap.release()
        return None

    def _open(self) -> cv2.VideoCapture:
        if self._is_url:
            candidates = self._url_candidates(self._source)
            for url in candidates:
                print(f"  → trying  {url}")
                cap = self._try_open(url)
                if cap is not None:
                    print(f"  ✓ connected: {url}")
                    self._source = url   # remember working URL for reconnects
                    return cap

            # Nothing worked — print actionable help before raising
            import socket
            ip_hint = self._source.split("/")[2].split(":")[0]
            reachable = False
            try:
                sock = socket.create_connection((ip_hint, 8080), timeout=2)
                sock.close()
                reachable = True
            except OSError:
                pass

            print("\n" + "=" * 60)
            print("  Could not connect to the phone stream.")
            print(f"  IP {ip_hint}:8080 is {'reachable ✓' if reachable else 'NOT reachable ✗'}")
            print()
            print("  Checklist:")
            print("  1. Phone and PC on the same Wi-Fi network")
            print("  2. 'IP Webcam' app → tap 'Start server' → note the IP")
            print(f"  3. Edit SOURCE in main() with that IP (currently: {ip_hint})")
            print("  4. Windows Firewall: allow Python on private networks")
            print()
            print("  Quick test: open in browser →  http://<phone-IP>:8080/video")
            print("=" * 60 + "\n")
            raise RuntimeError(f"All stream URLs failed for {self._source}")

        else:
            # Local webcam — DirectShow on Windows is faster
            backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self._source, backend)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FPS, 60)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self._source}")
            return cap

    # ── thread body (auto-reconnect on network drop) ──
    def run(self) -> None:
        consecutive_fails = 0
        while not self._stop.is_set():
            ok, frame = self._cap.read()
            if ok:
                consecutive_fails = 0
                with self._lock:
                    self._frame = frame
            else:
                consecutive_fails += 1
                if self._is_url and consecutive_fails > 10:
                    # Network dropped — try to reconnect
                    print("Stream lost — reconnecting …", end="\r")
                    self._cap.release()
                    time.sleep(1.0)
                    try:
                        self._cap = self._open()
                        consecutive_fails = 0
                        print("Stream restored.              ")
                    except RuntimeError:
                        pass

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def stop(self) -> None:
        self._stop.set()
        self._cap.release()


# ── Detector ──────────────────────────────────────────────────────────────────

class AnomalyDetector:

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        half: bool | None = None,
    ):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"Loading model … {path}")
        self.model = YOLO(str(path))

        self.conf_threshold = conf_threshold
        self.imgsz          = imgsz

        # Device & precision
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.half   = (self.device == "cuda") if half is None else half

        print(f"Device : {self.device}  |  FP16 : {self.half}  |  imgsz : {self.imgsz}")

        # Warm-up: eliminate first-inference overhead
        print("Warming up …")
        dummy = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
        for _ in range(3):
            self.model.predict(
                dummy, imgsz=imgsz, conf=conf_threshold,
                device=self.device, half=self.half, verbose=False,
            )
        print("Ready.\n")

        # FPS rolling average — O(1) append/pop
        self._times: deque[float] = deque(maxlen=30)
        self.fps = 0.0

        # Frame-skip state
        self.skip_n     = 0   # process 1 out of every (skip_n+1) frames
        self._counter   = 0
        self._cache     = None  # last inference result

    # ── inference ──
    def _infer(self, frame: np.ndarray):
        return self.model.predict(
            frame,
            imgsz=self.imgsz,
            conf=self.conf_threshold,
            device=self.device,
            half=self.half,
            verbose=False,
        )[0]

    # ── drawing ──
    def _draw(self, frame: np.ndarray, results) -> tuple[np.ndarray, int, int]:
        out   = frame.copy()
        boxes = results.boxes
        masks = results.masks

        if boxes is None or len(boxes) == 0:
            return out, 0, 0

        n_total   = len(boxes)
        n_anomaly = 0

        # Segmentation masks (filled, semi-transparent)
        if masks is not None:
            overlay = out.copy()
            for i, mdata in enumerate(masks.data):
                cls   = int(boxes.cls[i])
                color = CLASS_COLORS[cls % len(CLASS_COLORS)]
                m     = mdata.cpu().numpy()
                m     = cv2.resize(m, (frame.shape[1], frame.shape[0]))
                overlay[m > 0.5] = color
            cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)

        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf  = float(box.conf[0])
            cls   = int(box.cls[0])
            name  = self.model.names.get(cls, f"cls{cls}")
            color = CLASS_COLORS[cls % len(CLASS_COLORS)]

            n_anomaly += 1  # all detected classes are anomalies

            # Bounding box
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"{name} {conf:.2f}"
            (lw, lh), bl = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(out, (x1, y1 - lh - bl - 4),
                          (x1 + lw + 2, y1), color, -1)
            cv2.putText(out, label, (x1 + 1, y1 - bl),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                        COLORS["text"], 1, cv2.LINE_AA)

        return out, n_total, n_anomaly

    def _draw_hud(self, frame: np.ndarray,
                  n_det: int, n_anom: int) -> np.ndarray:
        is_pass = n_anom == 0
        status  = "PASS" if is_pass else f"FAIL  ({n_anom} anomal{'y' if n_anom==1 else 'ies'})"
        s_color = COLORS["pass"] if is_pass else COLORS["fail"]

        # Semi-transparent dark panel
        overlay = frame.copy()
        cv2.rectangle(overlay, (8, 8), (290, 155), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        lines = [
            (f"FPS   {self.fps:5.1f}", COLORS["text"]),
            (f"imgsz {self.imgsz}  skip {self.skip_n}", COLORS["text"]),
            (f"conf  {self.conf_threshold:.2f}", COLORS["text"]),
            (f"dets  {n_det}", COLORS["text"]),
            (f"device {self.device.upper()} {'FP16' if self.half else 'FP32'}", COLORS["text"]),
            (f"► {status}", s_color),
        ]

        y = 32
        for text, color in lines:
            cv2.putText(frame, text, (18, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
            y += 22

        # Key hints at bottom-right
        h, w = frame.shape[:2]
        hints = "[q]uit  [s]ave  [+/-]conf  [i]mgsz  [f]skip"
        cv2.putText(frame, hints, (w - 410, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 120, 120), 1, cv2.LINE_AA)
        return frame

    def _update_fps(self, elapsed: float) -> None:
        self._times.append(elapsed)
        avg = sum(self._times) / len(self._times)
        self.fps = 1.0 / avg if avg > 0 else 0.0

    # ── main loop ──
    def run_camera(self, source: int | str = 0,
                   width: int = 1280, height: int = 720) -> None:

        label = f"camera {source}" if isinstance(source, int) else source
        print(f"Opening {label} …")
        grabber = FrameGrabber(source, width, height)
        grabber.start()

        # Wait for the first frame (up to 3 s)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            ok, _ = grabber.read()
            if ok:
                break
            time.sleep(0.05)
        else:
            grabber.stop()
            raise RuntimeError("Camera did not produce a frame within 3 s.")

        print("Camera ready.")
        print("Keys : [q] quit  [s] save  [+/-] confidence  [i] imgsz 640↔320  [f] frame-skip")

        frame_count = 0
        WIN = "DefectVision — Real-time  [q: quit]"
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

        try:
            while True:
                t0 = time.time()

                ok, frame = grabber.read()
                if not ok or frame is None:
                    time.sleep(0.001)
                    continue

                # Frame-skip: reuse last result for skipped frames
                self._counter += 1
                if self._cache is None or self._counter % (self.skip_n + 1) == 0:
                    self._cache = self._infer(frame)

                annotated, n_det, n_anom = self._draw(frame, self._cache)
                annotated = self._draw_hud(annotated, n_det, n_anom)

                self._update_fps(time.time() - t0)

                cv2.imshow(WIN, annotated)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("s"):
                    fname = f"capture_{frame_count:04d}.jpg"
                    cv2.imwrite(fname, annotated)
                    print(f"Saved → {fname}")
                elif key in (ord("+"), ord("=")):
                    self.conf_threshold = min(0.95, self.conf_threshold + 0.05)
                    print(f"conf → {self.conf_threshold:.2f}")
                elif key in (ord("-"), ord("_")):
                    self.conf_threshold = max(0.05, self.conf_threshold - 0.05)
                    print(f"conf → {self.conf_threshold:.2f}")
                elif key == ord("i"):
                    self.imgsz = 320 if self.imgsz != 320 else 640
                    print(f"imgsz → {self.imgsz}")
                elif key == ord("f"):
                    self.skip_n = (self.skip_n + 1) % 5
                    print(f"skip → {self.skip_n}  (1 inference / {self.skip_n+1} frames)")

                frame_count += 1

        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            grabber.stop()
            cv2.destroyAllWindows()
            print(f"Done — {frame_count} frames  |  avg FPS {self.fps:.1f}")


# ── entry point ───────────────────────────────────────────────────────────────

def _pick_source() -> int | str:
    """Interactive menu — returns the chosen camera source."""
    print("=" * 60)
    print("  DefectVision Pro — Real-time Anomaly Detection")
    print("=" * 60)
    print()
    print("  Select camera source:")
    print()
    print("  [0]  PC webcam  (built-in or USB)")
    print("  [1]  iPhone  (Wi-Fi — Larix Broadcaster app)")
    print("  [2]  Android (Wi-Fi — IP Webcam app)")
    print()

    while True:
        choice = input("  Your choice (0 / 1 / 2): ").strip()

        # ── PC webcam ─────────────────────────────────────────────────────────
        if choice == "0":
            print()
            return 0

        # ── iPhone via Larix Broadcaster ──────────────────────────────────────
        if choice == "1":
            print()
            print("  ── iPhone setup (Larix Broadcaster) ──────────────────")
            print("  1. Install 'Larix Broadcaster' from the App Store (free)")
            print("  2. Open app → Settings → Connections → + → RTSP → Server")
            print("     Set port to 5540 → Save")
            print("  3. Tap the red Record button to start streaming")
            print("  ──────────────────────────────────────────────────────")
            print()
            ip_raw = input(
                "  Enter your iPhone IP (shown in Settings → Wi-Fi → ⓘ)\n"
                "  e.g.  192.168.1.67\n"
                "  > "
            ).strip()

            if not ip_raw:
                print("  No IP entered — falling back to PC webcam.\n")
                return 0

            ip = ip_raw.split(":")[0]   # strip port if user added it
            url = f"rtsp://{ip}:5540/ch0"
            print(f"\n  Source : {url}")
            print()
            return url

        # ── Android via IP Webcam ─────────────────────────────────────────────
        if choice == "2":
            print()
            print("  ── Android setup (IP Webcam) ─────────────────────────")
            print("  1. Install 'IP Webcam' from the Play Store (free)")
            print("  2. Open app → scroll to bottom → Start server")
            print("  3. Note the IP shown on screen (e.g. 192.168.1.87:8080)")
            print("  ──────────────────────────────────────────────────────")
            print()
            manual = input(
                "  Enter IP shown in the app, or press Enter to auto-scan\n"
                "  e.g.  192.168.1.87   or   192.168.1.87:8080\n"
                "  > "
            ).strip()

            if manual:
                host = manual if ":" in manual else f"{manual}:8080"
                url  = f"http://{host}/video"
                print(f"\n  Source : {url}")
                print()
                return url

            # Auto-scan
            print("\n  Scanning local network for IP Webcam on port 8080 …")
            found = discover_ip_webcam(port=8080)
            if found:
                print(f"  ✓ Found : {found}")
                return found

            print("  No IP Webcam found on port 8080.")
            print("  Make sure 'Start server' is tapped in the app.")
            retry = input("\n  Try again? (y/n): ").strip().lower()
            if retry != "y":
                print("  Falling back to PC webcam.\n")
                return 0
            continue

        print("  Please enter 0, 1 or 2.")


def _find_weights() -> str:
    """Return the first available .pt file from the weights/ directory."""
    base = Path(__file__).parent.parent / "weights"
    for name in ("best_m.pt", "best_s.pt", "best.pt"):
        p = base / name
        if p.exists() and p.stat().st_size > 1_000_000:
            return str(p)
    candidates = sorted(base.glob("*.pt"))
    candidates = [p for p in candidates if p.stat().st_size > 1_000_000]
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError(f"No valid .pt weights found in {base}")


def main():
    MODEL_PATH = _find_weights()
    CONF       = 0.25
    IMGSZ      = 640        # 320 → max speed | 640 → balanced | 800 → accuracy
    WIDTH      = 1280
    HEIGHT     = 720

    SOURCE = _pick_source()

    print()
    print(f"  Model  : {MODEL_PATH}")
    print(f"  imgsz  : {IMGSZ}   conf : {CONF}")
    print("=" * 60 + "\n")

    try:
        detector = AnomalyDetector(MODEL_PATH, conf_threshold=CONF, imgsz=IMGSZ)
        detector.run_camera(source=SOURCE, width=WIDTH, height=HEIGHT)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Expected weights at: {MODEL_PATH}")
    except Exception:
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
