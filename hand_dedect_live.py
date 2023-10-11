from ultralytics import YOLO
import cv2
from ultralytics.utils.plotting import Annotator
import torch
import numpy as np
import os
import torch.utils

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class Hand_dedector:
    def __init__(self) -> None:

        # [Tr] eğitilmiş modeli yükleme [En] loading the trained model
        self.model = YOLO("runs/detect/train/weights/best.pt")
        self.cap = cv2.VideoCapture(0)  # [Tr] kamera [En] camera
        self.rcords = []  # [Tr] "rcords = rounded cords" yuvarlanmış kordinatlar [En] Rounded cords
        self.h = 480
        self.w = 640

    def get_cropped_frame(self):
        _, frame = self.cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            source=frame, save=False, stream=True, show=False)
        self.cords = []
        # [Tr] tahmin edilen kutunun kordinatlarını alma.[En] get the coordinates of the predicted box.
        for r in results:

            annotator = Annotator(frame)

            boxes = r.boxes
            for box in boxes:
                b = box.xyxy[0]
                self.cords = b.tolist()
                self.rcords = [round(num) for num in self.cords]
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        # [Tr] [Tr]Resmi kırpma işlemi [En] Crop image
        if self.rcords:
            y1 = max(self.rcords[1]-40, 0)
            y2 = min(self.rcords[3]+40, self.h)
            x1 = max(self.rcords[0]-40, 0)
            x2 = min(self.rcords[2]+40, self.w)
            # [Tr] Resmi kırpın
            cropped_img = frame[y1:y2, x1:x2]
            # cropped_img = frame[self.rcords[1]-40:self.rcords[3] +
            #                     40, self.rcords[0]-40:self.rcords[2]+40]
            self.rcords.clear()
        else:
            cropped_img = frame
        return cropped_img, frame
