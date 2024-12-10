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

def inference_image(img_path, output_path, detector):
    
    if isinstance(img_path, np.ndarray):
        img = img_path
    else:
        img = cv2.imread(img_path)

    for _ in range(1):
        ta = time.time()
        bboxes, kpss = detector.detect(img, 0.3, input_size=(640, 640))
        tb = time.time()
        print(f'Inference time: {(tb - ta)} s'  )
    
    print(img_path, bboxes.shape)
    if kpss is not None:
        print(kpss.shape)
    
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i]
        x1, y1, x2, y2, _ = bbox.astype(np.int_)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        if kpss is not None:
            kps = kpss[i]
            for kp in kps:
                kp = kp.astype(np.int_)
                cv2.circle(img, tuple(kp), 1, (0, 0, 255), 2)
    
    filename = os.path.basename(img_path)
    path_output = os.path.join(output_path, f"output_{filename}")
    print('output:', path_output)
    cv2.imwrite(path_output, img)
    
    
def main(args):
    
    detector = SCRFD(model_file=cons.MODEL_SCRFD_PATH)
    detector.prepare(-1)
    
    if os.path.isdir(args.image_path):
        img_paths = [os.path.join(args.image_path, img_name) for img_name in os.listdir(args.image_path) if img_name.endswith(('.jpg', '.jpeg', '.png'))]
    else:
        img_paths = [args.image_path]

    os.makedirs(args.output_path, exist_ok=True)

    for img_path in img_paths:
        inference_image(img_path, args.output_path, detector)
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detect objects in images")
    parser.add_argument("--image_path", default=cons.IMAGE_TETS_1, help="Path to the image or folder containing images")
    parser.add_argument("--output_path", default=cons.OUTPUT_IMAGE_PATH, help="Path to save the output images")
    args = parser.parse_args()
    main(args)
    


