import mediapipe as mp
from mediapipe import solutions

import cv2
import time
import numpy as np


class PoseDetection:
    def __init__(self):
        
        self.result = mp.tasks.vision.HandLandmarkerResult
        self.landmarker = mp.tasks.vision.HandLandmarker
        self.create_landmarker()
        
    def update_result(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.result = result
        
    def create_landmarker(self):
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        BaseOptions = mp.tasks.BaseOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path="./pose_landmarker_lite.task"),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=self.update_result)
        
        self.landmarker = PoseLandmarker.create_from_options(options)
        
    def detect_async(self, frame):
        # convert np frame to mp.Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = frame)
        #detection
        self.landmarker.detect_async(image=mp_image, timestamp_ms = int(time.time() * 1000))
        
    def close(self):
        self.landmarker.close()
        
        
        

        