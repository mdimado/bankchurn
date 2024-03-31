from ultralytics import  YOLO
import cv2

model = YOLO("/Users/mohammedimaduddin/Desktop/bankchurn/bankchurn/backend/best.pt")
source = "/Users/mohammedimaduddin/Desktop/bankchurn/bankchurn/backend/testimages/A22_jpg.rf.f02ad8558ce1c88213b4f83c0bc66bc8.jpg"
model.predict(source)