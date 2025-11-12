
# cutting.py  ────────────────
import time
start_time = time.perf_counter()

import os
import cv2, mediapipe as mp, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

class ImageIO:
    """Unicode-safe image read / write."""
    @staticmethod
    def read(path: Path):
        buf = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    @staticmethod
    def write(path: Path, img, ext=".jpg"):
        ok, buf = cv2.imencode(ext, img)
        if not ok:
            raise RuntimeError("imencode fail")
        buf.tofile(str(path))

class FaceDetector:
    """MediaPipe BlazeFace wrapper."""
    def __init__(self, model_path: Path, min_conf=0.6):
        binary = model_path.read_bytes()
        opts   = mp.tasks.vision.FaceDetectorOptions(
            base_options=mp.tasks.BaseOptions(model_asset_buffer=binary),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            min_detection_confidence=min_conf,
        )
        self._detector = mp.tasks.vision.FaceDetector.create_from_options(opts)

    def detect(self, bgr):
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self._detector.detect(mp_img).detections

class FaceCropPipeline:
    """폴더 단위 얼굴 크롭 파이프라인."""
    def __init__(self, src_dir: Path, out_dir: Path,
                 detector: FaceDetector, margin: float = 0.25):
        self.src_dir, self.out_dir = src_dir, out_dir
        self.detector, self.margin = detector, margin
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def _crop_one(self, img_path: Path) -> int:
        img = ImageIO.read(img_path)
        if img is None:
            print(f"[SKIP] read fail {img_path}")
            return 0

        h, w, _ = img.shape
        dets = self.detector.detect(img)
        for idx, d in enumerate(dets, 1):
            bb = d.bounding_box
            x1 = int(max(bb.origin_x - bb.width *  self.margin, 0))
            y1 = int(max(bb.origin_y - bb.height * self.margin, 0))
            x2 = int(min(bb.origin_x + bb.width  * (1+self.margin), w))
            y2 = int(min(bb.origin_y + bb.height * (1+self.margin), h))
            face = img[y1:y2, x1:x2]

            # ★ 얼굴 크기 256x256으로 통일
            face = cv2.resize(face, (256, 256))

            out = self.out_dir / f"{img_path.stem}_face{idx}.jpg"
            ImageIO.write(out, face)
        return len(dets)

    def run(self, patterns=("*.jpg", "*.png"), workers=None):
        files = [p for pat in patterns for p in self.src_dir.glob(pat)]
        files.sort()
        workers = workers or min(os.cpu_count(), 8)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            counts = list(pool.map(self._crop_one, files))
        print(f"[{self.src_dir.name}] 완료: {sum(counts)} face(s) saved from {len(files)} image(s) → {self.out_dir}/")

# ──────────── 실행 스크립트 ────────────
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    detector = FaceDetector(ROOT / "models" / "blaze_face_short_range.tflite")

    targets = [("twice", "twice_faces"), ("stuff", "stuff_faces")]

    for src_folder, out_folder in targets:
        pipeline = FaceCropPipeline(
            src_dir = ROOT / src_folder,
            out_dir = ROOT / out_folder,
            detector= detector,
            margin  = 0.25,
        )
        pipeline.run()

    end_time = time.perf_counter()
    print(f"Total time: {end_time - start_time:.2f} s")

