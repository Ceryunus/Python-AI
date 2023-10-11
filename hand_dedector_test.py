from ultralytics import YOLO
import cv2
import torch
import os
from ultralytics.utils.plotting import Annotator

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

model = YOLO("runs/detect/train/weights/best.pt")

results=model.predict(source=0,show=True,save=False,stream=True)


for result in results:
    boxes = result.boxes 
    print("boxses",boxes)
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs  
print(results)
