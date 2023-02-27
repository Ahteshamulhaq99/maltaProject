import cv2
import torch
import datetime
import os
import time
from models.common import DetectMultiBackend
from utils.general import (LOGGER, check_img_size, non_max_suppression, scale_boxes, 
                                  check_imshow, xyxy2xywh, increment_path)
from utils.torch_utils import select_device, time_sync
from utils.plots import Annotator, colors
from deep_sort.utils.parser import get_config
from deep_sort.deep_sort import DeepSort
from video_stream import WebcamVideoStream
import numpy as np
from threading import Thread
from multiprocessing import Process
# from database import Database
# from exception_reid import *
# from send_pdf import *
# from urllib.request import urlopen
import uuid
import time
from detect_person import detect
from trackers.multi_tracker_zoo import create_tracker


class PersonTracking(object):

    def __init__(self) -> None:
        self.prev_day=datetime.datetime.today().date()
        self.curr_day=datetime.datetime.today().date()
        self.device,self.half = 'cpu', False
        self.model = DetectMultiBackend('yolov5n.pt')
    
    # def start_stream(self):
    #     cam_list=['rtsp://admin:disruptlab54321@192.168.0.6:554/Streaming/channels/1']
    #     for cam,path in enumerate(cam_list):
    #         Process(target=self.detect(path,cam+2,)).start()

    #     # Process(target=self.detect('rtsp://admin:disruptlab54321@192.168.0.2:554/Streaming/channels/1',1)).start()
    #     # Process(target=self.detect('rtsp://admin:disruptlab54321@192.168.0.4:554/Streaming/channels/1',2),args=('rtsp://admin:disruptlab54321@192.168.0.4:554/Streaming/channels/1',2,)).start()
    #     # self.detect('rtsp://admin:disruptlab54321@192.168.0.2:554/Streaming/channels/1',1)

                        

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        shape = im.shape[:2]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:
            r = min(r, 1.0)
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        if auto:
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)
        elif scaleFill:
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
        dw /= 2
        dh /= 2
        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        return im, ratio, (dw, dh)


    def create_folder(self,parent_path):
        today = datetime.datetime.now().strftime('%d-%m-%y')
        parent_path = os.path.join(parent_path, today).replace("\\","/") 
        if not os.path.exists(parent_path):
            os.makedirs(parent_path)
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
            os.makedirs(path)
            return path,True
        else:
            return path,False


    def check_date(self):
        if self.prev_day!=self.curr_day:
            self.check_shop=True
            if self.internet_on()==True:
                download_report(self.prev_day)
                self.prev_day=self.curr_day
            

    def internet_on(self):
        try:
            urlopen('http://google.com', timeout=5)
            return True
        except Exception as err: 
            return False
            

    def check_person(self,frame):
        _,val=detect(self.model,self.device, 416, frame,self.half,0.35)
        if len(val)==0:
            return True
        else:
            return False
            
            

    def detect(self,cam_path,cam_id):
        # db=Database()


        tracking_method = 'bytetrack'
        tracking_config="trackers/bytetrack/configs/bytetrack.yaml"
        reid_weights= 'osnet_x0_25_msmt17.pt'
        half=False
        device="cuda"
        byteTracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)

        # Initialize                                                                         
        half=True
        device = select_device('')
        half &= device.type != 'cpu'
        # Load model
        device = select_device(device)
        # model = DetectMultiBackend('crowdhuman_yolov5m.pt', device=device, dnn=True)
        model = torch.hub.load('ultralytics/yolov5',"custom", 'crowdhuman_yolov5m.pt')
        model.classes=1
        model.conf=0.15
      
        names=model.names

        customers =[]
        live_id_list =[]
        dtime = dict()
        dwell_time = dict()
        time_wait_counter = dict()
        img_val=dict()
        entry_time=dict()
        exit_time=dict()
        bbox=dict()
        person_in_count=0
        prev_path=''
        path=''
        total_dwell=0
        average_dwell=''
        path,val=self.create_folder('cam1')
        cap = WebcamVideoStream(src=cam_path).start()
        count=0
        absent_check=False
        person_check=False
        absent_counter=0
        black_out=True
        check_black=True
        bo_start_time=datetime.datetime.now()
        while True:
            count+=1
            im0 = cap.read()
            im1=im0.copy()
            im2=im0.copy()
            if im0 is None:
                continue
            annotator = Annotator(im0, line_width=4, pil=not ascii)

            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            ret,thresh = cv2.threshold(gray,70,255,0)
            black_pixels=cv2.countNonZero(thresh) 
            if black_pixels < 4000 and black_out == False:
                black_out= True
                bo_start_time=datetime.datetime.now()
                
            elif black_pixels > 4000 and black_out == True:
                black_out = False
                bo_end_time=datetime.datetime.now()
                # db.update_blackout(cam_id,datetime.datetime.today().date(),datetime.datetime.now(),bo_start_time,bo_end_time)
                print('Blackout Alert Pushed to db')

            if cam_id==2:
                if count%3==0:
                    t1=time.time()
                    result=self.check_person(im1)
                    if result==True:
                        absent_counter+=1
                        # print('Absent Counter: ',absent_counter)
                        if absent_counter==500:
                            absent_counter=0
                        if absent_counter>=30:
                            person_check=True
                            if absent_check==True:
                                # db.update_absent(2,datetime.datetime.today().date(),datetime.datetime.now(),True)
                                print("Absent Data Pushed to db")
                                absent_check=False
                                absent_counter=0
                    else:
                        absent_check=True
                        absent_counter=0
                        # print("Person Found")
                        if person_check==True:
                            # db.update_absent(2,datetime.datetime.today().date(),datetime.datetime.now(),False)
                            print("Present Found Data Pushed to db")
                            person_check=False
                    count=0
            
            self.curr_day=datetime.datetime.today().date()

            live_id_list =[]

            ##### STORING CUSTOMER AND STAFF TIME FOR PDF REPORT #####
            if cam_id==1:
                prev_path=path
                path,val=self.create_folder('cam1')
            if val == False:
                curr_date=datetime.datetime.now()
            elif val==True:
                if cam_id==1:
                    if person_in_count>0:
                        average_dwell=int((total_dwell/person_in_count))
                        # db.insert_count(int(cam_id),self.curr_day,str(prev_path[14:]),int(person_in_count),int(average_dwell),curr_date)
                        total_dwell=0
                        person_in_count=0

            # Inference
            pred = model(im0)
            # print(pred)
            outputs = byteTracker.update(torch.as_tensor(np.array(pred.xyxy[0].cpu())), im0)
            # Apply NMS
            # pred = non_max_suppression(pred, 0.5, 0.25,[0])

            # Process detections
            for det in outputs:  # detections per image
                
                # if det is not None and len(det):
                #     det[:, :4] = scale_boxes(
                #         img.shape[2:], det[:, :4], im0.shape).round()
                #     xywhs = xyxy2xywh(det[:, 0:4])
                #     confs = det[:, 4]
                #     clss = det[:, 5]

                    # pass detections to deepsort
                    # outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    
                    
                    # draw boxes for visualization
                if len(outputs) > 0:
                    for j, output in enumerate(outputs):
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]
                        x1 = int(output[0])
                        y1 = int(output[1])
                        x2 = int(output[2])
                        y2 = int(output[3])

                        #### Saving 10 images of each ID of staff's camera ####
                        if cam_id==2:  
                            try:         
                                width=x2-x1
                                if width> 20:
                                    uid = uuid.uuid1()
                                    if not os.path.exists(f"m2mresult/{id}"):
                                        os.makedirs(f"m2mresult/{id}")                               
                                    if len(os.listdir(f"m2mresult/{id}")) < 10:
                                        rr=int(width/5)
                                        cv2.imwrite(f"m2mresult/{id}/{uid.hex}.jpg",im2[y1-rr:y2+rr,x1-rr:x2+rr])

                                    if not os.path.exists(f"check_img/{id}"):
                                        os.makedirs(f"check_img/{id}")                               
                                    if len(os.listdir(f"check_img/{id}")) < 10:
                                        rr=int(width/5)
                                        cv2.imwrite(f"check_img/{id}/{uid.hex}.jpg",im2[y1-rr:y2+rr,x1-rr:x2+rr])       
                            except:
                                pass

                        live_id_list.append(id) ##### Appending ids which are currently in the frame into the list #####
                        
                        if id not in customers: ##### Checking if the that id is present in the customer's list or not #####  
                            customers.append(id) ##### If id not present in the list, append that id in the list #####
                            bbox[id]=[x1,y1,x2,y2] #### Bounding box coordinates being saved in the dictionary for that corresponding id #####
                            entry_time[id]=datetime.datetime.now() ##### Entry time for that id saved in the dictionary #####
                            dtime[id] = datetime.datetime.now()
                            dwell_time[id] = 0 ### dwell time initialized for that id #####

                            ### SENDING STAFF DATA ###
                            if cam_id==2:
                                prev_path=path
                                path,val=self.create_folder('cam2')
                                # db.insert_count_staff(int(cam_id),int(dwell_time[id]),self.curr_day,str(prev_path[14:]),int(id),datetime.datetime.now())
                                cropimg=im0.copy()
                                cropimg = cropimg[bbox[id][1]:bbox[id][3],bbox[id][0]:bbox[id][2]]
                            
                        else:
                            curr_time = datetime.datetime.now() #### If id already present in the customer's list #####
                            old_time = dtime[id] #### Assigning old time of that id #####
                            time_diff = curr_time - old_time ##### Calculating current time and prev time difference for dwell time #####
                            dtime[id] = datetime.datetime.now() ##### Changing old time to current time #####
                            sec = time_diff.total_seconds() ##### Converting time diff to seconds #####
                            dwell_time[id] += sec ##### Updating dwell time of that id #####

                        c = int(cls)
                        text = "Time: {} seconds".format( int(dwell_time[id]))
                        label = f'{id} {names[c]} {text}'
                        if id not in time_wait_counter.keys(): ##### Id if not present in time wait counter's list #####
                            time_wait_counter[id]=0 #### Initialize counter for that id #####

                        for c_id in customers: ##### Iterating over ids present in the customer's list #####
                            if c_id not in live_id_list: ##### If that particular id is not currently in the live list/frame #####
                                time_wait_counter[c_id]+=1 ##### Start waiting timer for that id #####
                                if time_wait_counter[c_id] >=10: ##### If counter greater than this threshold process of deleting that id #####
                                    if cam_id==1:
                                        if int (dwell_time[c_id])<=120:#### If counter is less than this threshold, conitnue and don't remove id #####
                                            continue
                                    if cam_id==2:
                                        if int (dwell_time[c_id])<=1:#### If counter is less than this threshold, conitnue and don't remove id #####
                                            continue

                                    ###### PARTICULAR ID DELETED #####
                                    newleft='{} left and at time {} seconds from camera {}'.format(c_id,int (dwell_time[c_id]),cam_id)  
                                    exit_time[c_id]=datetime.datetime.now() ##### Exit time for that id #####

                                    if cam_id==1:
                                        total_dwell+=int(dwell_time[c_id])
                                        person_in_count+=1
                                
                                    
                                    # if cam_id==2:
                                    #     db.update_dwell(int(cam_id),int(dwell_time[c_id]),c_id)

                                    
                                    del entry_time[c_id] ##### Delete entry time for that particular id #####
                                    del exit_time[c_id] ##### Delete exit time for that particular id #####
                                    del dwell_time[c_id] ##### Delete dwell time for that particular id #####
                                    del bbox[c_id] ##### Delete bboxes for that particular id #####
                                    print(newleft)
                                    customers.remove(c_id) ##### Remove that id from customer's list #####
                            
                            else:
                                time_wait_counter[c_id]=0

                        annotator.box_label(bboxes, label, color=colors(c, True))
                # else:
                #     deepsort.increment_ages()

                im0 = annotator.result()    
                im0=cv2.resize(im0,(1080,920))
                cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                cv2.imshow('Image', im0)
                if cv2.waitKey(1) == ord('q'):  
                    raise StopIteration



if __name__ == '__main__':
    P1=PersonTracking()
    src={1:'rtsp://admin:disruptlab54321@192.168.0.6:554/stream1'}

    for st in src: 
        p1 = Process(target=P1.detect(src[st],st+1)).start()