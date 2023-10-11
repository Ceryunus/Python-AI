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

        self.model = YOLO("runs/detect/train/weights/best.pt")#eğitilmi modeli yükleme
        self.cap = cv2.VideoCapture(0)#kamera
        self.rcords = []#"rounded cords" yuvarlanmış kordinatlar
        self.h=480
        self.w=640
    def get_cropped_frame(self):
        _, frame = self.cap.read()

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            source=frame, save=False, stream=True, show=False)
        self.cords = []
        # tahmin edilen kutunun kordinatlarını alma.
        for r in results:

            annotator = Annotator(frame)

            boxes = r.boxes
            for box in boxes:
                #    left   top   right    buttom
                # get box coordinates in (top, left, bottom, right) format
                b = box.xyxy[0]
                self.cords = b.tolist()
                self.rcords = [round(num) for num in self.cords]
                c = box.cls
                annotator.box_label(b, self.model.names[int(c)])

        # Resmi kırpma işlemi
        if self.rcords:
                        # Resmin boyutlarını alın
            

            # Kırpılacak bölgenin koordinatlarını hesaplayın ve sınırları kontrol edin
            y1 = max(self.rcords[1]-40, 0)
            y2 = min(self.rcords[3]+40, self.h)
            x1 = max(self.rcords[0]-40, 0)
            x2 = min(self.rcords[2]+40, self.w)
            # Resmi kırpın
            cropped_img = frame[y1:y2, x1:x2]
            # cropped_img = frame[self.rcords[1]-40:self.rcords[3] +
            #                     40, self.rcords[0]-40:self.rcords[2]+40]
            self.rcords.clear()
        else:
            cropped_img = frame
        return cropped_img, frame
