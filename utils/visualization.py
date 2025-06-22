import cv2
import numpy as np
import os

def generate_video_output(output_path, video_path, keypoints_list, fps=30):
    """
    Overlays keypoints on original video frames and saves the result.
    
    Parameters:
    - output_path (str): Path to save the generated video.
    - video_path (str): Original video file.
    - keypoints_list (List[np.ndarray]): List of (num_keypoints, 2) arrays for each frame.
    - fps (int): Frames per second for the output video.
    """


    print(f"[INFO] Processing video: {video_path}")  # Fixed here

    # Check if video_path is valid
    if not isinstance(video_path, str) or not os.path.exists(video_path):
        raise ValueError(f"Invalid video path: {video_path}")
    
    # Check if output_path is a valid directory
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    # Get frame size
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(keypoints_list):
            break

        keypoints = keypoints_list[frame_idx]
        
        # Overlay keypoints on the frame
        for x, y in keypoints:
            if not np.isnan(x) and not np.isnan(y):
                cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[✓] Output video saved at {output_path}")
