from PoseDetection import PoseDetection
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

def draw_landmarks_on_image(rgb_image, detection_result):
  try:
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image
  except:
      return rgb_image

if __name__ == "__main__":
    
    model_path = "./pose_landmarker_heavy.task"
    pose_landmarker = PoseDetection()
    
    cam = cv2.VideoCapture(0)
    
    while True:
        
        ret, frame = cam.read()
        
        if ret:
            
            pose_landmarker.detect_async(frame)
            
            frame = draw_landmarks_on_image(frame, pose_landmarker.result)
            
            cv2.imshow('frame', frame)
            cv2.waitKey(1)
