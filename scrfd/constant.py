from pydantic import BaseModel
import os

class Constant(BaseModel):
    MODEL_SCRFD_PATH: str = 'public/models/scrfd_500m_bnkps_shape640x640.onnx'
    IMAGE_ROOT: str = 'public/images/'
    IMAGE_TETS_1: str = os.path.join(IMAGE_ROOT, "image.jpg")
    VIDEO_ROOT: str = 'public/videos/'
    VIDEO_TEST_1: str = os.path.join(VIDEO_ROOT, 'videov.mp4') # 4-3 cho MUUU 
    OUTPUT_IMAGE_PATH = "public/images/output/"
    OUTPUT_VIDEO_PATH = "public/videos/output/"

cons = Constant()
print("Constant: ", cons)