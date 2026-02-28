import cv2
import numpy as np
import os
import json

# ==============================
# CONFIG
# ==============================

INPUT_FOLDER = "cctv"

MIN_CONTOUR_AREA = 1000        # ignore tiny movement
NO_MOTION_BUFFER_SEC = 1       # motion must stop for 1 sec
IGNORE_INITIAL_SEC = 2         # let background model learn

# ==============================
# PROCESS VIDEO FUNCTION
# ==============================

def process_video(video_path):

    print(f"\nProcessing: {video_path}")

    cap = cv2.VideoCapture(video_path)

    # ----- SAFETY CHECK 1 -----
    if not cap.isOpened():
        print("❌ Cannot open video (corrupted). Skipping.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)

    # ----- SAFETY CHECK 2 -----
    if fps == 0:
        print("❌ Invalid FPS. Skipping.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = video_path.replace(".mp4", "_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Background subtractor (MODEL)
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

        # let background model stabilize
        if frame_count < IGNORE_FRAMES:
            bg.apply(frame)
            continue

        # grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # reduce vibration noise
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # apply background subtraction
        fg_mask = bg.apply(gray)

        # binary cleanup
        _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8)
        )

        contours, _ = cv2.findContours(
            mask,
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

                        # write saved motion frames
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
    import os
    # ==========================
    # REPLACE ORIGINAL VIDEO
    # ==========================

    processed_path = output_path   # already created earlier

    # check processed file exists
    if os.path.exists(processed_path):

        # remove original video
        #os.remove(video_path)

        # rename processed -> original name
        #os.rename(processed_path, video_path)
        #instead of remove and and rename we use replace to do it in a single step
        os.replace(processed_path, video_path)

        print("✔ Original replaced with compressed video")
    else:
        print("❌ Processed video not found, skipping replacement")

        # save log
        log_path = video_path.replace(".mp4", "_log.json")
        with open(log_path, "w") as f:
            json.dump(events, f, indent=4)

        print(f"Finished: {video_path}")
        print(f"Events detected: {len(events)}")




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