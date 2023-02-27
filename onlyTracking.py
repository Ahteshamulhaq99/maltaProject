import cv2
 

import torch  
import numpy as np
import os
import uuid
from video_stream import WebcamVideoStream
from trackers.multi_tracker_zoo import create_tracker
  
import time







cam='rtsp://admin:disruptlab54321@192.168.0.4:554/stream1'
# cam=r"output.mp4"
# cap = cv2.VideoCapture(cam)
cap = WebcamVideoStream(src=cam)
cap.start()     

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.hub.load('ultralytics/yolov5',"custom", 'crowdhuman_yolov5m.pt')
# model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)


# print("nnnnnn",model.names)
model.classes=1
tracking_method = 'bytetrack'
tracking_config="trackers/bytetrack/configs/bytetrack.yaml"
reid_weights= 'osnet_x0_25_msmt17.pt'
half=False
byteTracker = create_tracker(tracking_method, tracking_config, reid_weights, device, half)



while True:
    t1=time.time()
    img = cap.read()
    img2=img.copy()

    results = model(img)
    print(results.xyxy[0].cpu())

    tracked = byteTracker.update(torch.as_tensor(np.array(results.xyxy[0].cpu())), img)
    # print(tracked)
    for boxes in tracked:
        # print(boxes)
        x1,y1,x2,y2,id,_,_=boxes
        

        cv2.rectangle(img,(int(x1), int(y1)), (int(x2),int(y2) ), (255, 0, 0), thickness=2)
        cv2.putText(img, str(id), (int(x1), int(y1)), cv2.FONT_HERSHEY_COMPLEX, fontScale=2, color=(0, 255,0))

    cv2.imshow("onlytrack", img)
    t2=time.time()
    
    print(1/(t2-t1))
    
    
    if cv2.waitKey(1) & 0xFF == 27:
        break
cv2.destroyAllWindows()
  