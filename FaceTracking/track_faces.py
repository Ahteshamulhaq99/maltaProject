import cv2
import torch
import datetime
from utils.general import xyxy2xywh, get_angle
from utils.torch_utils import select_device
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
import numpy as np
from multiprocessing import Process
from threading import Thread
import detect_face
from video_stream import WebcamVideoStream
from queue import Queue
import math
import os
from PIL import Image
import datetime

class FaceTracking(object):

    def __init__(self):
        self.camera_threads = []
        self.qcount = Queue(maxsize=100)
        cfg = get_config()
        cfg.merge_from_file("./deep_sort/configs/deep_sort.yaml")
        self.deepsort = DeepSort('mobilenetv2_x1_0', max_dist=cfg.DEEPSORT.MAX_DIST,max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)
        # Initialize
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("device: ", self.device)
        self.model = detect_face.load_model('yolov5s-face.pt', self.device)
        self.start_stream()
        


    def trackFaces(self, deepsort, confs, xywhs, img):
        clss = torch.as_tensor([0]*len(xywhs))
        xywhs = torch.as_tensor(xywhs)
        confs = torch.as_tensor(confs)

        if len(xywhs) > 0:
            outputs = deepsort.update(
                xywhs.cpu(), confs.cpu(), clss.cpu(), img)
        else:
            deepsort.increment_ages()
        return outputs

    
    def start_stream(self):
        Cameras={"cam2":['rtsp://admin:hik12345@192.168.1.65:554/Streaming/channels/1']}
        
        for cam in Cameras.keys():
            cam_id = cam[3:]
            cap = WebcamVideoStream(src=Cameras[cam][0])
            self.camera_threads.append([cap, cam_id])

        for camera_thread in self.camera_threads:
            camera_thread[0].start()
            Thread(target=self.detect, args=(camera_thread[0],camera_thread[1],)).start()
        Thread(target=self.counting()).start()

    def create_folder(self,parent_path):
        today = datetime.datetime.now().strftime('%d-%m-%y')
        parent_path = os.path.join(parent_path, today).replace("\\","/") 
        if not os.path.exists(parent_path):
            os.mkdir(parent_path)
        start_hour = str(datetime.datetime.now().strftime('%I%p'))
        if start_hour=='11AM':
            end_hour = '12PM'
        elif start_hour=='11PM':
            end_hour = '12AM'
        elif start_hour=='12AM':
            end_hour = '01AM'
        elif start_hour=='12PM':
            end_hour = '01PM'
        else:
            end_hour = str(int(start_hour[:2])+1) + '' + str(start_hour[-2:])
        path = os.path.join(parent_path, str(start_hour + '-' + end_hour)).replace("\\","/") 
        if not os.path.exists(path):
            os.mkdir(path)
        return path


    def detect(self,cap2,cam_id):
        count=0
        while True:
            im0 = cap2.read()
            if im0 is None:
                continue
            im0=im0[:-520,:]
            orig_img=im0.copy()
            # Inference
            tracked, xywhs, confs = [], [], []
            try:
                boxes, landmarks, confs = detect_face.detect_one(self.model, im0, self.device)
            except:
                pass
            angles = []
            landmarks_dic=[]
            for ind, landmark in enumerate(landmarks):
                box = boxes[ind]
                x1 = int(box[0])
                y1 = int(box[1])
                x2 = int(box[2])
                y2 = int(box[3])
                q1, r1, q2, r2 = landmark[:4]
                angle=get_angle((q2, r2),(q1, r1))
                angles.append(angle)
                landmarks_dic.append(landmark)  
            if len(boxes) > 0:
                xywhs = xyxy2xywh(torch.as_tensor(np.array(boxes)))
                try:
                    tracked = self.trackFaces(self.deepsort, np.array(confs), np.array(xywhs), im0)
                except:
                    pass
                img_angles = {}
                img_marks={}
                for x1, y1, x2, y2, id, _ in tracked:
                    if [x1, y1, x2, y2] in boxes:
                        index = boxes.index([x1, y1, x2, y2])
                        img_angles[id]= angles[index]
                        img_marks[id]=landmarks_dic[index]
                        cropped_img=orig_img[y1:y2,x1:x2]
                        cropped_img=cv2.resize(cropped_img,(416,416))
                        cropped_img = Image.fromarray(cropped_img)  
                        aligned_img=np.array(cropped_img.rotate(img_angles[id]))
                        path=self.create_folder('align_faces/cam2')
                        if aligned_img is not None:
                            count+=1
                            print(aligned_img)
                            cv2.imwrite(f'align_faces/cam2/{path}/{count}.jpg',aligned_img)

                # except Exception as e:
                #     print(f"Exception: {e}")
            self.qcount.put((cam_id, im0, tracked, xywhs, confs))


    def counting(self):
        while True:
            cam_id, im0, outputs, xywhs, confs=self.qcount.get()
            if im0 is None:
                continue
            if len(outputs) > 0:
                for output in outputs:
                    id = output[4]
                    x1 = output[0]
                    y1 = output[1]
                    x2 = output[2]
                    y2 = output[3]
                    cv2.rectangle(im0, (x1,y1), (x2,y2), (0,0,255), 2)
                    cv2.putText(im0, str(id), (x1+10,y1+10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                im0=cv2.resize(im0,(1080,720))
                if cam_id=='1':
                    cv2.namedWindow("Cam1", cv2.WINDOW_NORMAL)
                    cv2.imshow('Cam1',im0)
                    cv2.waitKey(1)
                if cam_id=='2':
                    cv2.namedWindow("Cam2", cv2.WINDOW_NORMAL)
                    cv2.imshow('Cam2',im0)
                    cv2.waitKey(1)

            
if __name__ == '__main__':
    FaceTracking()
