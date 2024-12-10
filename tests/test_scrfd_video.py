import os
import sys
sys.path.append(os.getcwd())
import cv2
import datetime
import numpy as np
import argparse
from scrfd.python.model import SCRFD
from scrfd.constant import cons
import time
from tqdm import tqdm

def inference_video(video_path, output_path, detector, fps=20):
    cap = cv2.VideoCapture(video_path)
    filename = os.path.basename(video_path)

    # Create a VideoWriter object to save the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    size = (int(frame_width), int(frame_height))
    out = cv2.VideoWriter(os.path.join(output_path, filename), fourcc, fps, size)
    
    count_frame = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total number of frames in the video
    start_time = time.time()
    fps = 0
    

    # Wrap the loop with tqdm for progress bar
    for _ in tqdm(range(total_frames), desc="Processing Frames"):
        # Capture a frame from the video
        count_frame += 1
        ret, frame = cap.read()

        if not ret:
            break

        # Detect faces in the frame
        bboxes, kpss = detector.detect(frame, 0.3, input_size=(640, 640))
        
        # Draw the bounding boxes and keypoints on the frame
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i]
            x1, y1, x2, y2, _ = bbox.astype(np.int_)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if kpss is not None:
                kps = kpss[i]
                for kp in kps:
                    kp = kp.astype(np.int_)
                    cv2.circle(frame, tuple(kp), 1, (0, 0, 255), 2)
        
        # Calculate FPS and display it on the frame
        end_time = time.time()
        
        if end_time - start_time > 1.0:
            fps = count_frame
            start_time = end_time
            count_frame = 0
            
        cv2.putText(frame, f"FPS: {fps}", (7, 70), cv2.FONT_HERSHEY_SIMPLEX, 3, (100, 255, 0), 3, cv2.LINE_AA)
        
        # Write the frame to the output video
        out.write(frame)

    # Release the video capture and video writer objects
    cap.release()
    out.release()
    
    
def main(args):
    detector = SCRFD(model_file=cons.MODEL_SCRFD_PATH)
    detector.prepare(-1)

    if os.path.isdir(args.video_path):
        img_paths = [os.path.join(args.video_path, img_name) for img_name in os.listdir(args.image_path) if img_name.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        img_paths = [args.video_path]
        
    os.makedirs(args.output_path, exist_ok=True)
    
    for img_path in img_paths:
        inference_video(img_path, args.output_path, detector)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in images")
    parser.add_argument("--video_path", default=cons.VIDEO_TEST_1, help="Path to the image or folder containing images")
    parser.add_argument("--output_path", default=cons.OUTPUT_VIDEO_PATH, help="Path to save the output images")
    args = parser.parse_args()
    main(args)
    


