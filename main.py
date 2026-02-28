import cv2
import numpy as np
import os
import json

folder_path = "cctv_folder"

videos = [f for f in os.listdir(folder_path) if f.endswith(".mp4")]

def process_video(video_path):

    cap = cv2.VideoCapture(video_path)
    bg = cv2.createBackgroundSubtractorMOG2()

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = video_path.replace(".mp4", "_processed.mp4")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    motion_segments = []
    motion_start = None
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        mask = bg.apply(frame)
        motion_pixels = np.sum(mask == 255)

        # Adjust sensitivity here
        if motion_pixels > 2000:
            if motion_start is None:
                motion_start = frame_number / fps

            out.write(frame)

        else:
            if motion_start is not None:
                motion_end = frame_number / fps
                motion_segments.append({
                    "start_seconds": motion_start,
                    "end_seconds": motion_end,
                    "duration_seconds": motion_end - motion_start
                })
                motion_start = None

        frame_number += 1

    # Handle case if motion continues till video ends
    if motion_start is not None:
        motion_end = frame_number / fps
        motion_segments.append({
            "start_seconds": motion_start,
            "end_seconds": motion_end,
            "duration_seconds": motion_end - motion_start
        })

    cap.release()
    out.release()

    # Save timestamps JSON
    log_path = video_path.replace(".mp4", "_log.json")
    with open(log_path, "w") as f:
        json.dump(motion_segments, f, indent=4)

    # Delete original file
    os.remove(video_path)

    print(f"Processed: {video_path}")

for video_name in videos:
    video_path = os.path.join(folder_path, video_name)
    process_video(video_path)

print("All videos processed successfully!")