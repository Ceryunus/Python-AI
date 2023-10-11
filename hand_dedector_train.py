from ultralytics import YOLO
import os

if __name__ == '__main__':
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    model = YOLO("yolov8n.yaml")  
    model.train(data="config.yaml", epochs=10)  # train the model
