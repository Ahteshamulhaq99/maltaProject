import cv2
import time
import threading
import numpy as np
import qimage2ndarray
from database import Database

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtGui import QIcon, QPixmap

class Footfall(object):
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

        # 'rtsp://admin:OMJANM@192.168.1.102:554/stream1'
        t1 = threading.Thread(target=self.display_video_stream, args=('static/staff.mp4', self.lbl_camera_staff,))
        t1.start()
        t2 = threading.Thread(target=self.display_video_stream, args=('static/customer.mp4', self.lbl_camera_customer,))
        t2.start()

        t3 = threading.Thread(target=self.lifo_staff, args=())
        t3.start()

        t4 = threading.Thread(target=self.lifo_customer, args=())
        t4.start()
        
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
    ui = Footfall()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

