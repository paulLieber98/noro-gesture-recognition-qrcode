import cv2 as cv
import mediapipe as mp
import time
import qrcode
import numpy as np
from typing import List
from mediapipe.tasks.python.core import base_options
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerResult, HandLandmarkerOptions
)
from mediapipe.tasks.python.vision import (
    ImageSegmenter, ImageSegmenterOptions
)

# -----------------------------
# Config
# -----------------------------
CAM_INDEX = 1
QR_URL = 'https://www.noro.co/'
QR_SHOW_SECONDS = 10
AREA_MIN = 1000  # min person box area to keep

# -----------------------------
# Globals (state across frames)
# -----------------------------
webcam = cv.VideoCapture(CAM_INDEX)
qrcode_is_shown = False
qrcode_shown_start_time = 0.0
should_show_qrcode = False

last_person_mask = None  # from segmentation callback

# -----------------------------
# Utilities
# -----------------------------
def generate_qr_bgr() -> np.ndarray:
    qr = qrcode.make(QR_URL).convert('RGB')
    return cv.cvtColor(np.array(qr), cv.COLOR_RGB2BGR)

def clamp_box(x, y, w, h, width, height):
    x = max(0, x); y = max(0, y)
    x2 = min(width, x + w); y2 = min(height, y + h)
    return x, y, x2 - x, y2 - y

def build_people_boxes_from_mask(mask: np.ndarray, frame_shape) -> List[dict]:
    # 1) resize to frame size
    h, w = frame_shape[:2]
    resized = cv.resize(mask, (w, h))
    # 2) threshold to binary
    binary = (resized > 0).astype(np.uint8) * 255
    # 3) find contours
    contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        x, y, bw, bh = cv.boundingRect(cnt)
        if bw * bh < AREA_MIN:
            continue
        x, y, bw, bh = clamp_box(x, y, bw, bh, w, h)
        if bw <= 0 or bh <= 0:
            continue
        boxes.append({'x': x, 'y': y, 'width': bw, 'height': bh})
    return boxes

def largest_box(boxes: List[dict]) -> dict | None:
    return max(boxes, key=lambda b: b['width'] * b['height']) if boxes else None

# -----------------------------
# Model callbacks
# -----------------------------
def on_hand_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    # If ONLY index finger up → trigger QR
    if not result.hand_landmarks:
        return
    finger_threshold = 0.05
    for hand_landmarks in result.hand_landmarks:
        if (
            hand_landmarks[8].y < hand_landmarks[5].y - finger_threshold and
            hand_landmarks[12].y > hand_landmarks[9].y + finger_threshold and
            hand_landmarks[16].y > hand_landmarks[13].y + finger_threshold and
            hand_landmarks[20].y > hand_landmarks[17].y + finger_threshold
        ):
            global should_show_qrcode
            should_show_qrcode = True
            return

def on_segmentation_result(result, output_image: mp.Image, timestamp_ms: int):
    # Update latest person mask for main loop
    if result.category_mask:
        global last_person_mask
        last_person_mask = result.category_mask.numpy_view()

# -----------------------------
# Model setup
# -----------------------------
BaseOptions = base_options.BaseOptions

hand_options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='hand_landmarker.task'),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=on_hand_result,
    num_hands=50
)

seg_options = ImageSegmenterOptions(
    base_options=BaseOptions(model_asset_path='selfie_segmenter.tflite'),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=on_segmentation_result,
    output_category_mask=True
)

# -----------------------------
# Main
# -----------------------------
with HandLandmarker.create_from_options(hand_options) as handmodel:
    with ImageSegmenter.create_from_options(seg_options) as segmenter:

        while True:
            ret, frame = webcam.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            # Prepare MediaPipe image
            frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Time for async calls
            timestamp_ms = int(time.time() * 1000)

            # Request async segmentation on full frame
            segmenter.segment_async(mp_image, timestamp_ms)

            # Build people boxes from latest mask (if available)
            people_boxes = []
            if last_person_mask is not None:
                people_boxes = build_people_boxes_from_mask(last_person_mask, frame.shape)

            # Run hand detection per-person (crop → upscale → RGB → detect)
            if people_boxes:
                for i, box in enumerate(people_boxes):
                    x1, y1 = box['x'], box['y']
                    x2, y2 = x1 + box['width'], y1 + box['height']
                    roi = frame[y1:y2, x1:x2]
                    if roi.size == 0:
                        continue
                    roi_up = cv.resize(roi, (frame.shape[1], frame.shape[0]))
                    roi_rgb = cv.cvtColor(roi_up, cv.COLOR_BGR2RGB)
                    mp_roi = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
                    handmodel.detect_async(mp_roi, timestamp_ms + i)  # unique ts per ROI
            else:
                # Fallback: allow gestures even if mask missing
                handmodel.detect_async(mp_image, timestamp_ms)

            # Display: main window is always full frame (mirrored)
            frame_main = cv.flip(frame, 1)
            cv.imshow('frame', frame_main)

            # Optional zoom window for largest person (visual only)
            if people_boxes:
                lb = largest_box(people_boxes)
                if lb:
                    lx, ly = lb['x'], lb['y']
                    lw, lh = lb['width'], lb['height']
                    zoom = cv.resize(frame[ly:ly+lh, lx:lx+lw], (frame.shape[1], frame.shape[0]))
                    cv.imshow('zoom', cv.flip(zoom, 1))
            else:
                cv.destroyWindow('zoom')

            # Show QR for a limited time if triggered
            global qrcode_is_shown, qrcode_shown_start_time, should_show_qrcode
            if should_show_qrcode:
                qrcode_is_shown = True
                qrcode_shown_start_time = time.time()
                cv.imshow('QR Code', generate_qr_bgr())
                should_show_qrcode = False

            if qrcode_is_shown and (time.time() - qrcode_shown_start_time) >= QR_SHOW_SECONDS:
                qrcode_is_shown = False
                qrcode_shown_start_time = 0.0
                cv.destroyWindow('QR Code')

            if cv.waitKey(1) == ord('q'):
                break

webcam.release()
cv.destroyAllWindows()