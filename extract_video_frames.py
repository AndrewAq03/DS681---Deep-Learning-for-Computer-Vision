# Andrew Aquino
# DS681 - Assingment 1

import cv2
import os

def extract_frames(video_path, output_folder, num_frames=1000):

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Total frames: {total_frames}, FPS: {fps}")

    # countes the frames
    frame_interval = max(1, total_frames // num_frames)
    frame_count = 0
    image_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        if frame_count % frame_interval == 0 and image_count < num_frames:
            image_name = f"frame_{image_count:04d}.jpg"
            image_path = os.path.join(output_folder, image_name)
            cv2.imwrite(image_path, frame)
            image_count += 1
        
        frame_count += 1

    cap.release()
    print(f"Successfully extracted {image_count} frames to '{output_folder}'.")


video_file = "tracking_assignment1.mp4" 
output_directory = "video_frames"
extract_frames(video_file, output_directory, num_frames=1000)