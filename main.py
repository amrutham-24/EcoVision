import cv2
import numpy as np
import os
import json

# ==============================
# CONFIG
# ==============================

INPUT_FOLDER = "cctv"

MIN_CONTOUR_AREA = 1000
NO_MOTION_BUFFER_SEC = 1.5
IGNORE_INITIAL_SEC = 2

LONG_DIFF_GAP = 3
THRESHOLD_VALUE = 140


# ==============================
# PROCESS VIDEO
# ==============================

def process_video(video_path):

    print(f"\nProcessing: {video_path}")

    cap = cv2.VideoCapture(video_path)

    # SAFETY CHECKS
    if not cap.isOpened():
        print("❌ Cannot open video. Skipping.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("❌ Invalid FPS. Skipping.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # create processed folder
    os.makedirs("processed", exist_ok=True)

    filename = os.path.basename(video_path)
    output_path = os.path.join("processed", filename)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # BACKGROUND MODEL
    bg = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    frame_count = 0
    motion_active = False
    motion_start = None
    no_motion_frames = 0

    events = []
    frame_buffer = []
    frame_history = []

    NO_MOTION_LIMIT = int(fps * NO_MOTION_BUFFER_SEC)
    IGNORE_FRAMES = int(fps * IGNORE_INITIAL_SEC)

    # ==========================
    # FRAME LOOP
    # ==========================

    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # background learning
        if frame_count < IGNORE_FRAMES:
            bg.apply(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        fg_mask = bg.apply(gray)

        _, mask = cv2.threshold(
            fg_mask,
            THRESHOLD_VALUE,
            255,
            cv2.THRESH_BINARY
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        # LONG FRAME DIFFERENCE (N-3)
        frame_history.append(gray)

        if len(frame_history) > LONG_DIFF_GAP:
            long_diff = cv2.absdiff(
                frame_history[-1],
                frame_history[-LONG_DIFF_GAP]
            )

            _, long_mask = cv2.threshold(
                long_diff,
                20,
                255,
                cv2.THRESH_BINARY
            )
        else:
            long_mask = np.zeros_like(gray)

        combined_mask = cv2.bitwise_or(mask, long_mask)

        contours, _ = cv2.findContours(
            combined_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        motion = False

        for c in contours:
            if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                motion = True
                break

        # ======================
        # MOTION LOGIC
        # ======================

        if motion:

            if not motion_active:
                motion_start = frame_count / fps
                motion_active = True

            no_motion_frames = 0
            frame_buffer.append(frame)

        else:

            if motion_active:
                no_motion_frames += 1

                if no_motion_frames > NO_MOTION_LIMIT:

                    motion_end = (frame_count - no_motion_frames) / fps
                    duration = motion_end - motion_start

                    if duration > 0:
                        events.append({
                            "start_seconds": round(motion_start, 3),
                            "end_seconds": round(motion_end, 3),
                            "duration_seconds": round(duration, 3)
                        })

                        for f in frame_buffer:
                            out.write(f)

                    frame_buffer = []
                    motion_active = False
                    no_motion_frames = 0

    # ==========================
    # CLEANUP
    # ==========================

    cap.release()
    out.release()

    # SAVE LOG
    log_path = os.path.join(
        "processed",
        filename.replace(".mp4", "_log.json")
    )

    with open(log_path, "w") as f:
        json.dump(events, f, indent=4)

    print(f"✔ Processed video saved → {output_path}")
    print(f"✔ Events detected: {len(events)}")


# ==============================
# MAIN
# ==============================

def main():

    if not os.path.exists(INPUT_FOLDER):
        print("❌ CCTV folder not found")
        return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp4")]

    if len(files) == 0:
        print("No videos found.")
        return

    for file in files:
        path = os.path.join(INPUT_FOLDER, file)
        process_video(path)


if __name__ == "__main__":
    main()