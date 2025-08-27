# count_traffic.py
from ultralytics import YOLO
import cv2
import numpy as np
from collections import defaultdict
import time

# ---------- Config ----------
VIDEO_SOURCE = "intersection.mp4" # or 0 for webcam
MODEL_PATH   = "yolov8m.pt"  # or "yolov8n.pt" for zero-shot
CONF_THRESH  = 0.15
IOU_THRESH   = 0.45
IMG_SIZE     = 640
TRACKER_CFG  = "botsort.yaml"  # built-in tracker config from Ultralytics
ALLOWED_LABELS = {
    "car", "truck", "bus", "motorcycle", "automobile", "van", "pickup", "vehicle", "suv", "minivan", "jeep"
}

# Counting line endpoints (x1,y1) -> (x2,y2). Adjust to your video resolution.
LINE_PTS     = ((100, 500), (1180, 500))
# Optional: restrict counting to a polygon region (ROI); set to None to disable
ROI_POLY     = None  # e.g., [(50,400),(1230,400),(1230,720),(50,720)]

# Counting box: (x1, y1, x2, y2)
# Move the box slightly more to the right
COUNT_BOX = (950, 350, 1600, 750)  # Shifted a bit further right

# ---------- Helpers ----------
def side_of_line(p, a, b):
    # cross product sign tells which side of the line AB the point P is on
    return (b[0]-a[0])*(p[1]-a[1]) - (b[1]-a[1])*(p[0]-a[0])

def crossed_line(p_prev, p_curr, a, b):
    s1 = side_of_line(p_prev, a, b)
    s2 = side_of_line(p_curr, a, b)
    return s1 * s2 < 0  # different sides

def point_in_poly(pt, poly):
    if poly is None: 
        return True
    return cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0

# ---------- Main ----------
def main():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO_SOURCE}")

    counts = defaultdict(int)     # per-class counts
    memory_centers = {}           # track_id -> last center point
    counted_ids = set()           # track_ids that have already crossed (optional single-line mode)

    a, b = LINE_PTS
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frames = 0

    # Stream results with tracker; persist=True keeps IDs consistent across frames
    results_gen = model.track(
        source=VIDEO_SOURCE,
        stream=True,
        tracker=TRACKER_CFG,
        conf=CONF_THRESH,
        iou=IOU_THRESH,
        imgsz=IMG_SIZE,
        verbose=False,
        persist=True
    )

    for result in results_gen:
        frame = result.orig_img
        frames += 1

        # Remove counting line
        # cv2.line(frame, a, b, (0, 255, 0), 2)  # <-- DELETE or comment out this line

        if ROI_POLY is not None:
            cv2.polylines(frame, [np.array(ROI_POLY, dtype=np.int32)], True, (255, 255, 255), 1)

        # Draw counting box
        x1b, y1b, x2b, y2b = COUNT_BOX
        cv2.rectangle(frame, (x1b, y1b), (x2b, y2b), (0, 255, 0), 2)

        if result.boxes is None or result.boxes.id is None:
            cv2.imshow("Traffic Counter", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break
            continue

        ids = result.boxes.id.cpu().numpy().astype(int)
        clses = result.boxes.cls.cpu().numpy().astype(int)
        xyxy = result.boxes.xyxy.cpu().numpy()

        for i, tid in enumerate(ids):
            x1, y1, x2, y2 = xyxy[i]
            cls_id = clses[i]
            label = result.names[int(cls_id)].lower()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Only count allowed vehicle classes
            if label not in ALLOWED_LABELS:
                continue

            # draw box + label + id
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(frame, f"{label} #{tid}", (int(x1), int(y1) - 6), font, 0.5, (255, 255, 255), 1)

            # Debug: print detected label
            print("Detected label:", label)

            # optional ROI filter (only count inside road area)
            if not point_in_poly((cx, cy), ROI_POLY):
                memory_centers[tid] = (cx, cy)
                continue

            # Count if any part of the bounding box enters the counting box
            box_in_count_box = (
                x1b < x2 and x2b > x1 and
                y1b < y2 and y2b > y1
            )
            prev = memory_centers.get(tid)
            memory_centers[tid] = (cx, cy)
            if tid not in counted_ids and box_in_count_box:
                counts[label] += 1
                counted_ids.add(tid)

        # Overlay totals
        y0 = 30
        cv2.putText(frame, "Counts:", (10, y0), font, 0.7, (0, 255, 255), 2)
        for k, v in sorted(counts.items()):
            y0 += 24
            cv2.putText(frame, f"{k}: {v}", (10, y0), font, 0.7, (0, 255, 255), 2)

        # FPS display
        elapsed = time.time() - start_time
        fps = frames / max(elapsed, 1e-6)
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-130, 30), font, 0.6, (0, 255, 0), 2)

        cv2.imshow("Traffic Counter", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Final counts:", dict(counts))

if __name__ == "__main__":
    main()

# Install required packages
# !pip install ultralytics opencv-python numpy
