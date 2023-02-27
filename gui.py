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
from datetime import datetime


import time
import threading
import numpy as np
import qimage2ndarray
from database import Database

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap


class PersonTracking(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1366, 820)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet("background-color: white;")

        self.staff_img = []
        self.staff_label = []

        self.customer_img = []
        self.customer_label = []

        self.staff_lifo = []
        self.customer_lifo = []

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.lbl_logo = QtWidgets.QLabel(self.centralwidget)
        self.lbl_logo.setGeometry(QtCore.QRect(40, 20, 271, 81))
        self.lbl_logo.setObjectName("lbl_logo")
        pixmap = QPixmap('./static/logo.png')
        self.lbl_logo.setPixmap(pixmap)

        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(22)
        
        self.lbl_camera_staff = QtWidgets.QLabel(self.centralwidget)
        self.lbl_camera_staff.setGeometry(QtCore.QRect(40, 130, 630, 410))
        self.lbl_camera_staff.setObjectName("lbl_camera_staff")
        self.lbl_camera_staff.setStyleSheet("border: 3px solid #1a3d6f;"
                                         " border-radius : 5px; ")
        
        self.lbl_camera_customer = QtWidgets.QLabel(self.centralwidget)
        self.lbl_camera_customer.setGeometry(QtCore.QRect(710, 130, 630, 410))
        self.lbl_camera_customer.setObjectName("lbl_camera_customer")
        self.lbl_camera_customer.setStyleSheet("border: 3px solid #1a3d6f;"
                                         " border-radius : 5px; ")
        
        self.lbl_label_staff = QtWidgets.QLabel(self.centralwidget)
        self.lbl_label_staff.setGeometry(QtCore.QRect(40, 720, 200, 70))
        self.lbl_label_staff.setObjectName("lbl_label_staff")
        self.lbl_label_staff.setFont(font)
        self.lbl_label_staff.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_label_staff.setStyleSheet("background-color: #1a3d6f; color : white;"
                                   "text-align: center;"
                                   "border-top: 3px solid #68686a;"
                                   "border-bottom: 3px solid #68686a;"
                                   "border-left: 3px solid #68686a;"
                                   "border-top-left-radius : 5px;"
                                   " border-top-right-radius : 0px; "
                                   "border-bottom-left-radius : 5px; "
                                   "border-bottom-right-radius : 0px")
        
        self.lbl_staff = QtWidgets.QLabel(self.centralwidget)
        self.lbl_staff.setGeometry(QtCore.QRect(240, 720, 430, 70))
        self.lbl_staff.setObjectName("lbl_staff")
        self.lbl_staff.setFont(font)
        self.lbl_staff.setStyleSheet("background-color: #1a3d6f; color : white;"
                                   "text-align: center;"
                                   "border-top: 3px solid #68686a;"
                                   "border-right: 3px solid #68686a;"
                                   "border-bottom: 3px solid #68686a;"
                                   "border-top-left-radius : 0px;"
                                   " border-top-right-radius : 5px; "
                                   "border-bottom-left-radius : 0px; "
                                   "border-bottom-right-radius : 5px")
        
        self.lbl_label_customer = QtWidgets.QLabel(self.centralwidget)
        self.lbl_label_customer.setGeometry(QtCore.QRect(710, 720, 220, 70))
        self.lbl_label_customer.setObjectName("lbl_label_customer")
        self.lbl_label_customer.setFont(font)
        self.lbl_label_customer.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_label_customer.setStyleSheet("background-color: #1a3d6f; color : white;"
                                   "text-align: center;"
                                   "border-top: 3px solid #68686a;"
                                   "border-bottom: 3px solid #68686a;"
                                   "border-left: 3px solid #68686a;"
                                   "border-top-left-radius : 5px;"
                                   " border-top-right-radius : 0px; "
                                   "border-bottom-left-radius : 5px; "
                                   "border-bottom-right-radius : 0px")
        
        self.lbl_customer = QtWidgets.QLabel(self.centralwidget)
        self.lbl_customer.setGeometry(QtCore.QRect(930, 720, 410, 70))
        self.lbl_customer.setObjectName("lbl_customer")
        self.lbl_customer.setFont(font)
        self.lbl_customer.setStyleSheet("background-color: #1a3d6f; color : white;"
                                   "text-align: center;"
                                   "border-top: 3px solid #68686a;"
                                   "border-right: 3px solid #68686a;"
                                   "border-bottom: 3px solid #68686a;"
                                   "border-top-left-radius : 0px;"
                                   " border-top-right-radius : 5px; "
                                   "border-bottom-left-radius : 0px; "
                                   "border-bottom-right-radius : 5px")

        ################################################################ STAFF Start

        ########## Button Left Staff START

        self.lbl_left_staff = QtWidgets.QLabel(self.centralwidget)
        self.lbl_left_staff.setGeometry(QtCore.QRect(40, 575, 100, 100))
        self.lbl_left_staff.setObjectName("lbl_left_staff")
        self.lbl_left_staff.setStyleSheet("")
        self.lbl_left_staff.setPixmap(QtGui.QPixmap("static/icon_left.png"))
        self.lbl_left_staff.mousePressEvent = self.move_left_staff
        
        ########## Button Left Staff END

        x_sticker = 120
        y_sticker_height = 560
        y_label_height = 657

        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        
        self.lbl_staff1 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_staff1.setGeometry(QtCore.QRect(x_sticker, y_sticker_height, 150, 100))
        self.lbl_staff1.setObjectName("lbl_staff1")
        self.lbl_staff1.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 5px;"
                                        "border-top-right-radius : 5px; "
                                        "border-bottom-left-radius : 0px; "
                                        "border-bottom-right-radius : 0px")
        self.lbl_staff1.setHidden(True)

        self.lbl_staff1_l = QtWidgets.QLabel(self.centralwidget)
        self.lbl_staff1_l.setGeometry(QtCore.QRect(x_sticker, y_label_height, 150, 40))
        self.lbl_staff1_l.setObjectName("lbl_staff1_l")
        self.lbl_staff1_l.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 0px;"
                                        "border-top-right-radius : 0px; "
                                        "border-bottom-left-radius : 5px; "
                                        "border-bottom-right-radius : 5px")
        self.lbl_staff1_l.setFont(font)
        self.lbl_staff1_l.setHidden(True)
        self.lbl_staff1_l.setText("")
        
        x_sticker += 160

        self.lbl_staff2 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_staff2.setGeometry(QtCore.QRect(x_sticker, y_sticker_height, 150, 100))
        self.lbl_staff2.setObjectName("lbl_staff2")
        self.lbl_staff2.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 5px;"
                                        "border-top-right-radius : 5px; "
                                        "border-bottom-left-radius : 0px; "
                                        "border-bottom-right-radius : 0px")
        self.lbl_staff2.setHidden(True)

        self.lbl_staff2_l = QtWidgets.QLabel(self.centralwidget)
        self.lbl_staff2_l.setGeometry(QtCore.QRect(x_sticker, y_label_height, 150, 40))
        self.lbl_staff2_l.setObjectName("lbl_staff2_l")
        self.lbl_staff2_l.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 0px;"
                                        "border-top-right-radius : 0px; "
                                        "border-bottom-left-radius : 5px; "
                                        "border-bottom-right-radius : 5px")
        self.lbl_staff2_l.setFont(font)
        self.lbl_staff2_l.setHidden(True)
        
        x_sticker += 160
        
        self.lbl_staff3 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_staff3.setGeometry(QtCore.QRect(x_sticker, y_sticker_height, 150, 100))
        self.lbl_staff3.setObjectName("lbl_staff3")
        self.lbl_staff3.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 5px;"
                                        "border-top-right-radius : 5px; "
                                        "border-bottom-left-radius : 0px; "
                                        "border-bottom-right-radius : 0px")
        self.lbl_staff3.setHidden(True)

        self.lbl_staff3_l = QtWidgets.QLabel(self.centralwidget)
        self.lbl_staff3_l.setGeometry(QtCore.QRect(x_sticker, y_label_height, 150, 40))
        self.lbl_staff3_l.setObjectName("lbl_staff3_l")
        self.lbl_staff3_l.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 0px;"
                                        "border-top-right-radius : 0px; "
                                        "border-bottom-left-radius : 5px; "
                                        "border-bottom-right-radius : 5px")
        self.lbl_staff3_l.setFont(font)
        self.lbl_staff3_l.setHidden(True)

        ########## Button Right Staff START

        self.lbl_right_staff = QtWidgets.QLabel(self.centralwidget)
        self.lbl_right_staff.setGeometry(QtCore.QRect(610, 575, 100, 100))
        self.lbl_right_staff.setObjectName("lbl_right_staff")
        self.lbl_right_staff.setStyleSheet("")
        self.lbl_right_staff.setPixmap(QtGui.QPixmap("static/icon_right.png"))
        self.lbl_right_staff.mousePressEvent = self.move_right_staff
        
        ########## Button Right Staff END

        ################################################################ STAFF END

        ################################################################ CUSTOMER START

        ########## Button Left Customer START

        self.lbl_left_customer = QtWidgets.QLabel(self.centralwidget)
        self.lbl_left_customer.setGeometry(QtCore.QRect(710, 575, 100, 100))
        self.lbl_left_customer.setObjectName("lbl_left_staff")
        self.lbl_left_customer.setStyleSheet("")
        self.lbl_left_customer.setPixmap(QtGui.QPixmap("static/icon_left.png"))
        self.lbl_left_customer.mousePressEvent = self.move_left_customer

        ########## Button Left Customer END

        x_sticker_customer = 790
        y_sticker_height_customer = 560
        y_label_height_customer = 657

        font = QtGui.QFont()
        font.setFamily("Arial Black")
        font.setPointSize(12)
        
        self.lbl_customer1 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_customer1.setGeometry(QtCore.QRect(x_sticker_customer, y_sticker_height_customer, 150, 100))
        self.lbl_customer1.setObjectName("lbl_customer1")
        self.lbl_customer1.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 5px;"
                                        "border-top-right-radius : 5px; "
                                        "border-bottom-left-radius : 0px; "
                                        "border-bottom-right-radius : 0px")
        self.lbl_customer1.setHidden(True)

        self.lbl_customer_l = QtWidgets.QLabel(self.centralwidget)
        self.lbl_customer_l.setGeometry(QtCore.QRect(x_sticker_customer, y_label_height_customer, 150, 40))
        self.lbl_customer_l.setObjectName("lbl_customer_l")
        self.lbl_customer_l.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 0px;"
                                        "border-top-right-radius : 0px; "
                                        "border-bottom-left-radius : 5px; "
                                        "border-bottom-right-radius : 5px")
        self.lbl_customer_l.setFont(font)
        self.lbl_customer_l.setHidden(True)
        
        x_sticker_customer += 160

        self.lbl_customer2 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_customer2.setGeometry(QtCore.QRect(x_sticker_customer, y_sticker_height_customer, 150, 100))
        self.lbl_customer2.setObjectName("lbl_customer2")
        self.lbl_customer2.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 5px;"
                                        "border-top-right-radius : 5px; "
                                        "border-bottom-left-radius : 0px; "
                                        "border-bottom-right-radius : 0px")
        self.lbl_customer2.setHidden(True)

        self.lbl_customer2_l = QtWidgets.QLabel(self.centralwidget)
        self.lbl_customer2_l.setGeometry(QtCore.QRect(x_sticker_customer, y_label_height_customer, 150, 40))
        self.lbl_customer2_l.setObjectName("lbl_customer2_l")
        self.lbl_customer2_l.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 0px;"
                                        "border-top-right-radius : 0px; "
                                        "border-bottom-left-radius : 5px; "
                                        "border-bottom-right-radius : 5px")
        self.lbl_customer2_l.setFont(font)
        self.lbl_customer2_l.setHidden(True)
        
        x_sticker_customer += 160
        
        self.lbl_customer3 = QtWidgets.QLabel(self.centralwidget)
        self.lbl_customer3.setGeometry(QtCore.QRect(x_sticker_customer, y_sticker_height_customer, 150, 100))
        self.lbl_customer3.setObjectName("lbl_customer3")
        self.lbl_customer3.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 5px;"
                                        "border-top-right-radius : 5px; "
                                        "border-bottom-left-radius : 0px; "
                                        "border-bottom-right-radius : 0px")
        self.lbl_customer3.setHidden(True)

        self.lbl_customer3_l = QtWidgets.QLabel(self.centralwidget)
        self.lbl_customer3_l.setGeometry(QtCore.QRect(x_sticker_customer, y_label_height_customer, 150, 40))
        self.lbl_customer3_l.setObjectName("lbl_customer3_l")
        self.lbl_customer3_l.setStyleSheet("border: 3px solid #1a3d6f;"
                                        "border-top-left-radius : 0px;"
                                        "border-top-right-radius : 0px; "
                                        "border-bottom-left-radius : 5px; "
                                        "border-bottom-right-radius : 5px")
        self.lbl_customer3_l.setFont(font)
        self.lbl_customer3_l.setHidden(True)

        ########## Button Right Staff START

        self.lbl_right_customer = QtWidgets.QLabel(self.centralwidget)
        self.lbl_right_customer.setGeometry(QtCore.QRect(1280, 575, 100, 100))
        self.lbl_right_customer.setObjectName("lbl_right_staff")
        self.lbl_right_customer.setStyleSheet("")
        self.lbl_right_customer.setPixmap(QtGui.QPixmap("static/icon_right.png"))
        self.lbl_right_customer.mousePressEvent = self.move_right_customer

        ########## Button Right Staff END

        ################################################################ CUSTOMER END

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1366, 23))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


        self.prev_day=datetime.today().date()
        self.curr_day=datetime.today().date()

        self.start_stream()

        # 'rtsp://admin:OMJANM@192.168.1.102:554/stream1'
        # t1 = threading.Thread(target=self.display_video_stream, args=('static/staff.mp4', self.lbl_camera_staff,))
        # t1.start()
        # t2 = threading.Thread(target=self.display_video_stream, args=('static/customer.mp4', self.lbl_camera_customer,))
        # t2.start()

        # t3 = threading.Thread(target=self.lifo_staff, args=())
        # t3.start()

        # t4 = threading.Thread(target=self.lifo_customer, args=())
        # t4.start()



    def lifo_staff(self):
        person_count = 0
        for i in range(1, 3):
            if len(self.staff_lifo)==3:
                self.staff_lifo.pop(0)
            self.staff_img.append(QtGui.QPixmap(f"person/person ({i})"))
            self.staff_label.append(f"Time: 00:{np.random.randint(low = 2, high = 60)}:00")
            self.staff_lifo.append(person_count)
            person_count+=1
            
            time.sleep(3)
            self.show_staff_lifo()
        
    def show_staff_lifo(self):
        lst_labels_image = (self.lbl_staff1, self.lbl_staff2, self.lbl_staff3)
        lst_label_dwell_time = (self.lbl_staff1_l, self.lbl_staff2_l, self.lbl_staff3_l)
        for idx, person_idx in enumerate(self.staff_lifo):
            lst_labels_image[idx].setPixmap(self.staff_img[person_idx])
            lst_label_dwell_time[idx].setText(self.staff_label[person_idx])

            lst_labels_image[idx].setHidden(False)
            lst_label_dwell_time[idx].setHidden(False)

    def lifo_customer(self):
        person_count = 0
        for i in range(5, 10):
            if len(self.customer_lifo)==3:
                self.customer_lifo.pop(0)
            self.customer_img.append(QtGui.QPixmap(f"person/person ({i})"))
            self.customer_label.append(f"Time: 00:{np.random.randint(low = 2, high = 60)}:00")
            self.customer_lifo.append(person_count)
            person_count+=1
            
            time.sleep(1.5)
            self.show_customer_lifo()
            print()
        
    def show_customer_lifo(self):
        lst_labels_image = (self.lbl_customer1, self.lbl_customer2, self.lbl_customer3)
        lst_label_dwell_time = (self.lbl_customer_l, self.lbl_customer2_l, self.lbl_customer3_l)
        for idx, person_idx in enumerate(self.customer_lifo):
            lst_labels_image[idx].setPixmap(self.customer_img[person_idx])
            lst_label_dwell_time[idx].setText(self.customer_label[person_idx])

            lst_labels_image[idx].setHidden(False)
            lst_label_dwell_time[idx].setHidden(False)

    def move_left_staff(self, event):
        person_length = len(self.staff_label)
        first_indx_lifo = self.staff_lifo[0]
        if person_length>0:
            if first_indx_lifo>-1:
                if person_length>first_indx_lifo:
                    self.staff_lifo.pop(-1)
                    self.staff_lifo.insert(0, first_indx_lifo-1)
                    self.show_staff_lifo()

    def move_right_staff(self, event):
        person_length = len(self.staff_label)
        last_indx_lifo = self.staff_lifo[-1]
        if person_length>0:
            if last_indx_lifo<(person_length-1):
                self.staff_lifo.pop(0)
                self.staff_lifo.append(last_indx_lifo+1)
                self.show_staff_lifo()

    def move_left_customer(self, event):
        person_length = len(self.customer_label)
        first_indx_lifo = self.customer_lifo[0]
        if person_length>0:
            if first_indx_lifo>-1:
                if person_length>first_indx_lifo:
                    self.customer_lifo.pop(-1)
                    self.customer_lifo.insert(0, first_indx_lifo-1)
                    self.show_customer_lifo()

    def move_right_customer(self, event):
        person_length = len(self.customer_label)
        last_indx_lifo = self.customer_lifo[-1]
        if person_length>0:
            if last_indx_lifo<(person_length-1):
                self.customer_lifo.pop(0)
                self.customer_lifo.append(last_indx_lifo+1)
                self.show_customer_lifo()

    def display_video_stream(self, path, image_label):
        while True:
            capture = cv2.VideoCapture(path)
            frame_count=0
            while True:
                if frame_count%7==0:
                    ret, frame = capture.read()
                    if ret:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (630, 410))
                        image = qimage2ndarray.array2qimage(frame) 
                        image_label.setPixmap(QPixmap.fromImage(image))
                    else:
                        time.sleep(0.2)
                        break
                frame_count+=1

    
    def start_stream(self):
        Process(target=self.detect,args=('rtsp://admin:hik12345@192.168.1.64:554/Streaming/channels/1',1,)).start()
        # Process(target=self.detect,args=('rtsp://admin:hik12345@192.168.1.65:554/Streaming/channels/1',2,)).start()


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
        return path


    # def check_date(self):
    #     if self.prev_day!=self.curr_day:
    #         return True
    #     else:
    #         return False


    def detect(self,path,cam_id):
        db=Database()
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file("deep_sort/configs/deep_sort.yaml")
        deepsort = DeepSort('osnet_x0_25',
                            max_dist=cfg.DEEPSORT.MAX_DIST,
                            max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                            max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
                            use_cuda=True)
        # Initialize                                                                         
        half=True
        device = select_device('')
        half &= device.type != 'cpu'
        # Load model
        device = select_device(device)
        model = DetectMultiBackend('yolov5l.pt', device=device, dnn=True)
        stride, names, pt, jit, _ = model.stride, model.names, model.pt, model.jit, model.onnx
        half &= pt and device.type != 'cpu'
        if pt:
            model.model.half() if half else model.model.float()
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
        prev_frame_time = 0
        new_frame_time = 0
        person_in_count=0

        cap = WebcamVideoStream(src=path).start()
        while True:
            im0 = cap.read()
            new_frame_time = time.time()
            if im0 is None:
                continue
            if cam_id==1:
                im0=im0[240:,:]
            img = self.letterbox(im0, 1024 , stride=32 , auto=True)[0]
            img = img.transpose((2, 0, 1))[::-1]
            img = np.ascontiguousarray(img)
            cropimg = im0.copy()
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            live_id_list =[]
            # Inference
            pred = model(img, augment=False)
            # Apply NMS
            pred = non_max_suppression(pred, 0.25, 0.2,[0])
            # Process detections
            for i, det in enumerate(pred):  # detections per image
                annotator = Annotator(im0, line_width=4, pil=not ascii)
                if det is not None and len(det):
                    det[:, :4] = scale_boxes(
                        img.shape[2:], det[:, :4], im0.shape).round()
                    xywhs = xyxy2xywh(det[:, 0:4])
                    confs = det[:, 4]
                    clss = det[:, 5]
                    # pass detections to deepsort
                    outputs = deepsort.update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                    # draw boxes for visualization
                    if len(outputs) > 0:
                        for j, (output, conf) in enumerate(zip(outputs, confs)):
                            bboxes = output[0:4]
                            id = output[4]
                            cls = output[5]
                            x1 = output[0]
                            y1 = output[1]
                            x2 = output[2]
                            y2 = output[3]  

                            live_id_list.append(id) ##### Appending ids which are currently in the frame into the list #####
                            
                            if id not in customers: ##### Checking if the that id is present in the customer's list or not #####  
                                customers.append(id) ##### If id not present in the list, append that id in the list #####
                                bbox[id]=[x1,y1,x2,y2] #### Bounding box coordinates being saved in the dictionary for that corresponding id #####
                                entry_time[id]=datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S") ##### Entry time for that id saved in the dictionary #####
                                dtime[id] = datetime.datetime.now()
                                dwell_time[id] = 0 ### dwell time initialized for that id #####
                                cropimg = cropimg[bbox[id][1]:bbox[id][3],bbox[id][0]:bbox[id][2]]
                                if cam_id==1:
                                    path=self.create_folder('cam1')
                                    cv2.imwrite(f'{path}/{id}.jpg',cropimg)
                                    img_val[int(id)]=f'cam1/{id}.jpg'
                                if cam_id==2:
                                    path=self.create_folder('cam2')
                                    cv2.imwrite(f'{path}/{id}.jpg',cropimg)
                                    img_val[int(id)]=f'cam2/{id}.jpg'
                                cropimg=im0.copy()

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
                                    if time_wait_counter[c_id] >=300: ##### If counter greater than this threshold process of deleting that id #####
                                        if int (dwell_time[c_id])<=10:#### If counter is less than this threshold, conitnue and don't remove id #####
                                            continue
                                        ###### PARTICULAR ID DELETED #####
                                        newleft='{} left and at time {} seconds from camera {}'.format(c_id,int (dwell_time[c_id]),cam_id)  
                                        exit_time[c_id]=datetime.datetime.now().strftime("%Y-%m-%d%H:%M:%S") ##### Exit time for that id #####
                                        person_in_count+=1
                                        ### SEND DATA TO DATABASE ###
                                        if cam_id==2:
                                            db.insert_staff(int(cam_id),int(c_id),img_val.get(int(c_id)),entry_time.get(int(c_id)),exit_time.get(int(c_id)),int(dwell_time.get(int( c_id))))
                                        if cam_id==1:
                                            db.insert_customer(int(cam_id),int(c_id),img_val.get(int(c_id)),entry_time.get(int(c_id)),exit_time.get(int(c_id)),int(dwell_time.get(int(c_id))),int(person_in_count))
                                        print("DATA PUSHED TO DB")
                                        del img_val[c_id] ##### Delete image path for that particular id #####
                                        del entry_time[c_id] ##### Delete entry time for that particular id #####
                                        del exit_time[c_id] ##### Delete exit time for that particular id #####
                                        del dwell_time[c_id] ##### Delete dwell time for that particular id #####
                                        del bbox[c_id] ##### Delete bboxes for that particular id #####
                                        print(newleft)
                                        customers.remove(c_id) ##### Remove that id from customer's list #####
                                else:
                                    time_wait_counter[c_id]=0

                            annotator.box_label(bboxes, label, color=colors(c, True))
                else:
                    deepsort.increment_ages()

                fps = 1/(new_frame_time-prev_frame_time)
                prev_frame_time = new_frame_time
                # print("Fps: ",fps)
                im0 = annotator.result()    
                im0=cv2.resize(im0,(1080,920))
                cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
                cv2.imshow('Image', im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Footfall"))
        self.lbl_label_staff.setText(_translate("MainWindow", "STAFF:"))
        self.lbl_staff.setText(_translate("MainWindow", "0"))
        self.lbl_customer.setText(_translate("MainWindow", "0"))
        self.lbl_label_customer.setText(_translate("MainWindow", "CUSTOMER:"))     

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = PersonTracking()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

