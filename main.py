import cv2
import numpy as np
import os
import json

INPUT_FOLDER = "cctv"

MIN_CONTOUR_AREA = 1000
NO_MOTION_BUFFER_SECONDS = 1
IGNORE_INITIAL_SECONDS = 2



def process_video(video_path):

    print(f"\nProcessing: {video_path}")

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error opening video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = video_path.replace(".mp4", "_processed.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    backSub = cv2.createBackgroundSubtractorMOG2(
        history=500,
        varThreshold=50,
        detectShadows=True
    )

    frame_count = 0
    motion_detected = False
    motion_start = None
    no_motion_frames = 0
    events = []

    NO_MOTION_THRESHOLD = int(fps * NO_MOTION_BUFFER_SECONDS)
    IGNORE_INITIAL_FRAMES = int(fps * IGNORE_INITIAL_SECONDS)

    frames_buffer = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count < IGNORE_INITIAL_FRAMES:
            backSub.apply(frame)
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        fg_mask = backSub.apply(gray)

        _, mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3),np.uint8))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion = False

        for c in contours:
            if cv2.contourArea(c) > MIN_CONTOUR_AREA:
                motion = True
                break

        if motion:
            if not motion_detected:
                motion_start = frame_count / fps
                motion_detected = True

            no_motion_frames = 0
            frames_buffer.append(frame)

        else:
            if motion_detected:
                no_motion_frames += 1

                if no_motion_frames > NO_MOTION_THRESHOLD:
                    motion_end = (frame_count - no_motion_frames) / fps
                    duration = motion_end - motion_start

                    if duration > 0:
                        events.append({
                            "start_seconds": round(motion_start, 3),
                            "end_seconds": round(motion_end, 3),
                            "duration_seconds": round(duration, 3)
                        })

                        for f in frames_buffer:
                            out.write(f)

                    frames_buffer = []
                    motion_detected = False
                    no_motion_frames = 0

    cap.release()
    out.release()

    log_path = video_path.replace(".mp4", "_log.json")
    with open(log_path, "w") as f:
        json.dump(events, f, indent=4)

    print(f"Finished: {video_path}")
    print(f"Events detected: {len(events)}")


def main():
    for file in os.listdir(INPUT_FOLDER):
        if file.endswith(".mp4"):
            process_video(os.path.join(INPUT_FOLDER, file))


if __name__ == "__main__":
    main()