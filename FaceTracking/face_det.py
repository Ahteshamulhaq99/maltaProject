import cv2
import torch
import detect_face
import os
import time

# class FaceDet:

#     def __init__(self):
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         print("Device: ", self.device)
#         self.model = detect_face.load_model('FaceTracking/yolov5s-face.pt', self.device)
#         self.crop_face()


def crop_face():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: ", device)
    model = detect_face.load_model('FaceTracking/yolov5s-face.pt', device)
    while True:
        print("Doing For Cam1")
        os.chdir('D:/VisionProject/PersonTracking/temp/cam1')
        for file in os.listdir():
            img=cv2.imread(file)
            boxes, landmarks, confs = detect_face.detect_one(
            model, img, device)
            if len(boxes)>0:
                for box in boxes:
                    img=img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    try:
                        cv2.imwrite("D:/VisionProject/PersonTracking/Faces/cam1/"+file,img)
                    except:pass
            os.remove(file)
            print("Processing Done")

        time.sleep(3)
        print("Doing For Cam2")
        os.chdir('D:/VisionProject/PersonTracking/temp/cam2')
        for file in os.listdir():
            img=cv2.imread(file)
            boxes, landmarks, confs = detect_face.detect_one(
            model, img, device)
            if len(boxes)>0:
                for box in boxes:
                    img=img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
                    try:
                        cv2.imwrite("D:/VisionProject/PersonTracking/Faces/cam2/"+file,img)
                    except:pass
            os.remove(file)
            print("Processing Done")

crop_face()