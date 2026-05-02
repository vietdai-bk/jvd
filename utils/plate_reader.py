import cv2
import numpy as np
import re
from typing import Optional, Dict, Tuple
from pathlib import Path


def _clean_plate_text(raw: str) -> str:
    if not raw:
        return ""
    cleaned = re.sub(r'[^A-Z0-9]', '', raw.upper())
    corrections = {
        'O': '0', 'Q': '0', 'D': '0', 'U': '0',
        'I': '1', 'L': '1', 'Z': '2', 'S': '5', 'B': '8'
    }
    for wrong, correct in corrections.items():
        cleaned = cleaned.replace(wrong, correct)
    
    match = re.search(r'([0-9]{2})([A-Z]{1,2})([0-9]{4,5})', cleaned)
    if match:
        return f"{match.group(1)}{match.group(2)}{match.group(3)}"

    return cleaned if len(cleaned) >= 6 else ""


def _enhance_plate(plate_img: np.ndarray) -> np.ndarray:
    if plate_img is None or plate_img.size == 0:
        return plate_img
    
    if len(plate_img.shape) == 3:
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = plate_img.copy()

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    
    return enhanced


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _four_point_transform(image: np.ndarray, pts: np.ndarray) -> np.ndarray:
    rect = _order_points(pts)
    tl, tr, br, bl = rect
    maxW = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    maxH = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if maxW <= 0 or maxH <= 0:
        return image
    dst = np.array([[0, 0], [maxW - 1, 0],
                    [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(image, M, (maxW, maxH))


def _rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def rectify_plate(plate_img: np.ndarray) -> np.ndarray:
    """
    Căn chỉnh biển số: thử perspective warp trước, fallback sang rotation.
    """
    if plate_img is None or plate_img.size == 0:
        return plate_img

    H, W = plate_img.shape[:2]
    img_area = H * W

    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best_cnt = max(contours, key=cv2.contourArea) if contours else None
    if best_cnt is not None:
        area_ratio = cv2.contourArea(best_cnt) / img_area
        if area_ratio > 0.7:
            rect = cv2.minAreaRect(best_cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            warped = _four_point_transform(plate_img, box.astype("float32"))
            if warped.size > 0:
                return warped

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180,
                            threshold=50, minLineLength=40, maxLineGap=10)
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            length = np.hypot(x2 - x1, y2 - y1)
            if abs(angle) < 30 and length > 30:
                angles.append(angle)
        if angles:
            best_angle = float(np.median(angles))
            if abs(best_angle) < 20:
                return _rotate_image(plate_img, best_angle - 2)

    return plate_img


class PlateReader:

    def __init__(self, config: dict):
        plate_cfg = config.get('plate', {})
        self._model_path = plate_cfg.get('model_path', '')
        self._conf = plate_cfg.get('conf', 0.4)
        self._cache_frames = plate_cfg.get('cache_frames', 30)
        self._padding = plate_cfg.get('padding', 5)
        self._pad_ratio = plate_cfg.get('pad_ratio', 0.15)
        self._debug = plate_cfg.get('debug', False)

        self._yolo = None
        self._ocr = None
        self._enabled = False

        self._cache: Dict[int, Tuple[str, int]] = {}

        self._init_models(plate_cfg)

    def _init_models(self, plate_cfg: dict):
        # Init YOLO plate detector
        if self._model_path and Path(self._model_path).exists():
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(self._model_path)
                print(f"[PlateReader] YOLO model loaded: {self._model_path}")
            except Exception as e:
                print(f"[PlateReader] Không load được YOLO: {e}")
        else:
            if self._model_path:
                print(f"[PlateReader] Không tìm thấy model: {self._model_path}")
            else:
                print("[PlateReader] Không có model_path, dùng heuristic crop")

        # Init PaddleOCR (Optimized Pipeline)
        try:
            from paddleocr import PaddleOCR
            
            ocr_kwargs = {
                "lang": plate_cfg.get('ocr_lang', 'en'),
                "show_log": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
                "text_detection_model_name": "PP-OCRv4_mobile_det",
                "text_recognition_model_name": "PP-OCRv4_mobile_rec",
            }
            
            if plate_cfg.get('det_model_dir'):
                ocr_kwargs["det_model_dir"] = plate_cfg['det_model_dir']
            if plate_cfg.get('rec_model_dir'):
                ocr_kwargs["rec_model_dir"] = plate_cfg['rec_model_dir']
            if plate_cfg.get('rec_char_dict_path'):
                ocr_kwargs["rec_char_dict_path"] = plate_cfg['rec_char_dict_path']
                
            # Cấu hình phần cứng
            ocr_kwargs["use_gpu"] = plate_cfg.get('use_gpu', False)
            if plate_cfg.get('use_onnx', False):
                ocr_kwargs["use_onnx"] = True

            self._ocr = PaddleOCR(**ocr_kwargs)
            print("[PlateReader] PaddleOCR (API Tối ưu) sẵn sàng")
            
        except ImportError:
            print("[PlateReader] PaddleOCR chưa cài: pip install paddleocr")
        except Exception as e:
            print(f"[PlateReader] Lỗi PaddleOCR: {e}")

        self._enabled = self._ocr is not None
        mode = "YOLO+OCR" if (self._yolo and self._enabled) else \
               "OCR-only (heuristic crop)" if self._enabled else "DISABLED"
        print(f"[PlateReader] Mode: {mode}")

    @property
    def enabled(self) -> bool:
        return self._enabled

    def read(self, track_id: int, raw_frame: np.ndarray,
             vehicle_bbox, frame_id: int) -> str:
        if not self._enabled:
            return ""

        cached = self._cache.get(track_id)
        if cached is not None:
            plate_text, last_frame = cached
            if frame_id - last_frame < self._cache_frames and plate_text:
                return plate_text

        h, w = raw_frame.shape[:2]
        x1 = max(0, int(vehicle_bbox.x1))
        y1 = max(0, int(vehicle_bbox.y1))
        x2 = min(w, int(vehicle_bbox.x2))
        y2 = min(h, int(vehicle_bbox.y2))
        vehicle_crop = raw_frame[y1:y2, x1:x2]
        if vehicle_crop.size == 0:
            return ""

        plate_raw = self._get_plate_crop(vehicle_crop)
        if plate_raw is None or plate_raw.size == 0:
            return ""

        plate_rectified = rectify_plate(plate_raw)
        if plate_rectified is None or plate_rectified.size == 0:
            plate_rectified = plate_raw

        ph, pw = plate_rectified.shape[:2]
        if ph < 32:
            scale = max(2, int(64 / ph))
            plate_scaled = cv2.resize(plate_rectified, (pw * scale, ph * scale),
                                      interpolation=cv2.INTER_CUBIC)
        else:
            plate_scaled = plate_rectified

        if self._debug:
            self._show_debug(track_id, vehicle_crop, plate_raw, plate_rectified, plate_scaled)

        plate_text = self._run_ocr(plate_scaled)
        if self._debug:
            print(f"[PlateReader DEBUG] ID={track_id} raw_ocr='{plate_text}'")

        self._cache[track_id] = (plate_text, frame_id)
        return plate_text

    def _show_debug(self, track_id: int, vehicle_crop: np.ndarray,
                    plate_raw: np.ndarray, plate_rectified: np.ndarray,
                    plate_final: np.ndarray):
        TARGET_H = 120

        def _resize_h(img, h=TARGET_H):
            if img is None or img.size == 0:
                return np.zeros((h, h * 3, 3), dtype=np.uint8)
            ih, iw = img.shape[:2]
            ratio = h / ih
            return cv2.resize(img, (max(1, int(iw * ratio)), h),
                              interpolation=cv2.INTER_CUBIC)

        def _label(img, text):
            out = img.copy()
            if len(out.shape) == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(out, (0, 0), (out.shape[1], 20), (0, 0, 0), -1)
            cv2.putText(out, text, (4, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)
            return out

        panels = [
            _label(_resize_h(vehicle_crop), "1-vehicle_crop"),
            _label(_resize_h(plate_raw), "2-plate_raw"),
            _label(_resize_h(plate_rectified), "3-rectified"),
            _label(_resize_h(plate_final), "4-final(scaled)"),
        ]

        sep = np.zeros((TARGET_H, 2, 3), dtype=np.uint8)
        row = panels[0]
        for p in panels[1:]:
            row = np.hstack([row, sep, p])

        win = f"PlateDebug ID={track_id}"
        cv2.imshow(win, row)
        cv2.waitKey(1)

    def _get_plate_crop(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        if self._yolo is not None:
            return self._yolo_crop(vehicle_crop)
        return self._heuristic_crop(vehicle_crop)

    def _yolo_crop(self, vehicle_crop: np.ndarray) -> Optional[np.ndarray]:
        try:
            results = self._yolo(vehicle_crop, verbose=False, conf=self._conf)
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                return self._heuristic_crop(vehicle_crop)

            confs = results[0].boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            bx1, by1, bx2, by2 = results[0].boxes.xyxy[best_idx].cpu().numpy().astype(float)

            bw = bx2 - bx1
            bh = by2 - by1
            px = max(self._padding, int(bw * self._pad_ratio))
            py = max(self._padding, int(bh * self._pad_ratio))

            bx1 = max(0, int(bx1) - px)
            by1 = max(0, int(by1) - py)
            bx2 = min(vehicle_crop.shape[1], int(bx2) + px)
            by2 = min(vehicle_crop.shape[0], int(by2) + py)

            crop = vehicle_crop[by1:by2, bx1:bx2]
            return crop if crop.size > 0 else self._heuristic_crop(vehicle_crop)
        except Exception:
            return self._heuristic_crop(vehicle_crop)

    def _heuristic_crop(self, vehicle_crop: np.ndarray) -> np.ndarray:
        h, w = vehicle_crop.shape[:2]
        y_start = int(h * 0.55)
        x_margin = int(w * 0.1)
        return vehicle_crop[y_start:h, x_margin:w - x_margin]
    
    
    def _run_ocr(self, plate_crop: np.ndarray) -> str:
        if plate_crop is None or plate_crop.size == 0:
            return ""
        enhanced = _enhance_plate(plate_crop)
        
        try:
            # Truyền cls=False vì không dùng model phân loại góc
            result = self._ocr.ocr(enhanced, cls=False)
            
            # Print debug để chắc chắn format nhận được
            # print(f"Raw OCR result: {result}") 

            if not result or not result[0]:
                return ""
                
            texts = []
            
            # Cấu trúc của result thường là: [  [ [box], (text, score) ], [ [box], (text, score) ]  ]
            # Ta duyệt qua các dòng trong result[0]
            for line in result[0]:
                # Kiểm tra an toàn: line phải là list/tuple có ít nhất 2 phần tử
                # Phần tử thứ 2 (line[1]) phải là tuple chứa (text, confidence)
                if isinstance(line, (list, tuple)) and len(line) >= 2:
                    text_tuple = line[1]
                    if isinstance(text_tuple, (list, tuple)) and len(text_tuple) > 0:
                        text = str(text_tuple[0]) # Ép kiểu về string để đảm bảo an toàn cho regex
                        
                        # Xóa các ký tự không cần thiết (dấu gạch ngang, chấm, khoảng trắng)
                        text = re.sub(r'[-. ]', '', text)
                        texts.append(text)
            
            if not texts:
                return ""

            raw_text = ''.join(texts)

            return _clean_plate_text(raw_text)
            
        except Exception as e:
            print(f"[PlateReader] OCR error: {e}")
            return ""
    

    def get_cached(self, track_id: int) -> str:
        cached = self._cache.get(track_id)
        return cached[0] if cached else ""

    def cleanup_stale(self, current_frame_id: int, max_stale: int = 300):
        stale = [tid for tid, (_, lf) in self._cache.items()
                 if current_frame_id - lf > max_stale]
        for tid in stale:
            del self._cache[tid]