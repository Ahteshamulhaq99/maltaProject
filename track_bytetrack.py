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
from database import Database
from exception_reid import *
from send_pdf import *
from trackers import *
from trackers.multi_tracker_zoo import create_tracker
from urllib.request import urlopen


class PersonTracking(object):

    def __init__(self) -> None:
        self.prev_day=datetime.datetime.today().date()
        self.curr_day=datetime.datetime.today().date()

    
    def start_stream(self):
        # Process(target=self.detect,args=('rtsp://admin:hik12345@192.168.1.64:554/Streaming/channels/1',1,)).start()
        # Process(target=self.detect,args=('rtsp://admin:hik12345@192.168.1.65:554/Streaming/channels/1',1,)).start()
        self.detect('rtsp://admin:hik12345@192.168.1.64:554/Streaming/channels/1',1)


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
            return path,True
        else:
            return path,False


    def check_date(self):
        if self.prev_day!=self.curr_day:
            if self.internet_on()==True:
                download_report(self.prev_day)
                self.prev_day=self.curr_day
            

    def internet_on(self):
        try:
            urlopen('http://google.com', timeout=5)
            return True
        except Exception as err: 
            return False


    def detect(self,cam_path,cam_id):
        db=Database()
        # initialize ByteTrackers
        tracking_method = 'bytetrack'
        tracking_config="trackers/bytetrack/configs/bytetrack.yaml"
        reid_weights= 'osnet_x0_25_msmt17.pt'
        half=False
        device=''
        byteTracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)

        # Initialize                                                                         
        half=True
        device = select_device('')
        half &= device.type != 'cpu'
        # Load model
        device = select_device(device)
        model = torch.hub.load('ultralytics/yolov5', "custom",path="v5m.pt")
        names = model.module.names if hasattr(model, 'module') else model.names


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

        cap = WebcamVideoStream(src=cam_path).start()
        while True:
            im0 = cap.read()
            if im0 is None:
                continue
            if cam_id==1:
                im0=im0[:-330,:]
            if cam_id==2:
                im0=im0[:-520,:]
            self.curr_day=datetime.datetime.today().date()
            self.check_date()
            live_id_list =[]
            pred = model(im0)
            outputs = byteTracker.update(torch.as_tensor(np.array(pred.xyxy[0].cpu())), im0)
            annotator = Annotator(im0, line_width=4, pil=not ascii)
            for result in outputs:
                bboxes = result[0:4]
                bboxx = [int(x) for x in bboxes]
                x1,y1,x2,y2=bboxx
                id=result[4]

                live_id_list.append(id) ##### Appending ids which are currently in the frame into the list #####
                
                if id not in customers: ##### Checking if the that id is present in the customer's list or not #####  
                    customers.append(id) ##### If id not present in the list, append that id in the list #####
                    bbox[id]=[x1,y1,x2,y2] #### Bounding box coordinates being saved in the dictionary for that corresponding id #####
                    entry_time[id]=datetime.datetime.now() ##### Entry time for that id saved in the dictionary #####
                    dtime[id] = datetime.datetime.now()
                    dwell_time[id] = 0 ### dwell time initialized for that id #####
                    cropimg=im0.copy()
                    cropimg = cropimg[bbox[id][1]:bbox[id][3],bbox[id][0]:bbox[id][2]]
                else:
                    curr_time = datetime.datetime.now() #### If id already present in the customer's list #####
                    old_time = dtime[id] #### Assigning old time of that id #####
                    time_diff = curr_time - old_time ##### Calculating current time and prev time difference for dwell time #####
                    dtime[id] = datetime.datetime.now() ##### Changing old time to current time #####
                    sec = time_diff.total_seconds() ##### Converting time diff to seconds #####
                    dwell_time[id] += sec ##### Updating dwell time of that id #####

                c = 0
                text = "Time: {} seconds".format( int(dwell_time[id]))
                label = f'{id} {names[c]} {text}'
                if id not in time_wait_counter.keys(): ##### Id if not present in time wait counter's list #####
                    time_wait_counter[id]=0 #### Initialize counter for that id #####

                for c_id in customers: ##### Iterating over ids present in the customer's list #####
                    if c_id not in live_id_list: ##### If that particular id is not currently in the live list/frame #####
                        time_wait_counter[c_id]+=1 ##### Start waiting timer for that id #####
                        if time_wait_counter[c_id] >=200: ##### If counter greater than this threshold process of deleting that id #####
                            if int (dwell_time[c_id])<=35:#### If counter is less than this threshold, conitnue and don't remove id #####
                                continue

                            ###### PARTICULAR ID DELETED #####
                            newleft='{} left and at time {} seconds from camera {}'.format(c_id,int (dwell_time[c_id]),cam_id)  
                            exit_time[c_id]=datetime.datetime.now() ##### Exit time for that id #####
                            total_dwell+=int(dwell_time[c_id])
                            person_in_count+=1

                            ##### STORING CUSTOMER AND STAFF TIME FOR PDF REPORT #####
                            path,val=self.create_folder('cam1')
                            img_val[int(c_id)]=f'cam1/{c_id}.jpg'
                            path,val=self.create_folder('cam2')
                            img_val[int(c_id)]=f'cam2/{c_id}.jpg'
                            if val==True:
                                average_dwell=str(round((total_dwell/person_in_count)/60,2)) + ' minutes'
                                db.insert_count(int(cam_id),self.curr_day,str(path[14:]),int(person_in_count),str(average_dwell))
                                person_in_count=0
                                total_dwell=0

                            ##### SEND DATA TO DATABASE ###
                            if cam_id==2:
                                db.insert_staff(int(cam_id),int(c_id),img_val.get(int(c_id)),entry_time.get(int(c_id)),exit_time.get(int(c_id)),int(dwell_time.get(int( c_id))))
                            if cam_id==1:
                                db.insert_customer(int(cam_id),int(c_id),img_val.get(int(c_id)),entry_time.get(int(c_id)),exit_time.get(int(c_id)),int(dwell_time.get(int(c_id))))
                            del img_val[c_id] ##### Delete image path for that particular id #####
                            del entry_time[c_id] ##### Delete entry time for that particular id #####
                            del exit_time[c_id] ##### Delete exit time for that particular id #####
                            del dwell_time[c_id] ##### Delete dwell time for that particular id #####
                            del bbox[c_id] ##### Delete bboxes for that particular id #####
                            print(f"Person Count in Cam{cam_id} is: ", person_in_count)
                            print(newleft)
                            customers.remove(c_id) ##### Remove that id from customer's list #####
                    else:
                        time_wait_counter[c_id]=0

                annotator.box_label(bboxes, label, color=colors(c, True))

            im0 = annotator.result()    
            im0=cv2.resize(im0,(1080,920))
            cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
            cv2.imshow('Image', im0)
            if cv2.waitKey(1) == ord('q'):  
                raise StopIteration



if __name__ == '__main__':
    PersonTracking().start_stream() 
