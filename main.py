from ultralytics import YOLO
from sort import *
import numpy as np
import cv2
import torch
import math
import time
import random
import os
import cvzone

# define preferrence
font = cv2.FONT_HERSHEY_PLAIN
font_scale = 1
font_thickness = 2
rect_thickness = 4
white_text = (255, 255, 255)
colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(20)]

# define processing device mode
device: str = "mps" if torch.backends.mps.is_available() else "cpu"

# define YOLO weights model 
model_path = os.path.join(".", "weights", "yolov9e.pt")
model = YOLO(model_path)
model.to(device) # set processing unit

# define video path
video_path = os.path.join(".", "videos", "thai-traffic.mp4")
output_path = os.path.join(".", "videos", "output.mp4")
# define video 
cap = cv2.VideoCapture(video_path)
        
# define window
cv2.namedWindow("Frame")

# mouse callback function
def show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Coordinates: x={x}, y={y}")
cv2.setMouseCallback("Frame", show_coordinates)

# define FPS time
previous_time = time.time()

# read video
success, frame = cap.read()
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), cap.get(cv2.CAP_PROP_FPS), (frame.shape[1], frame.shape[0]))

# define roi mask 
mask_path = os.path.join(".", "images", "traffic_mask.png")
mask = cv2.imread(mask_path)

# define tracker
tracker = Sort(max_age=99, min_hits=3, iou_threshold=0.5)
totalCount = []

line_couter = [402, 519, 865, 552]

# couter background
counter_overlay_path = os.path.join(".", "images", "car_counter_bg.png")
counter_overlay = cv2.imread(counter_overlay_path, cv2.IMREAD_UNCHANGED)
counter_overlay = cv2.resize(counter_overlay, None, fx=0.4, fy=0.4, interpolation=cv2.INTER_LINEAR)

# loop each frame
while True:
    success, frame = cap.read()
    
    if not success:
        break
    
    cvzone.overlayPNG(frame, counter_overlay, (frame.shape[1] - 300, frame.shape[0] - 100))
    
    roi_mask = cv2.bitwise_and(frame, mask)
    
    # display roi mask
    # frame = roi_mask
    
    results = model(roi_mask, stream=True)
    
    # get current time and calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time)
    previous_time = current_time
    
    # display Processing Mode
    # cv2.putText(frame, f"[Processing Mode: {device} ] [FPS: {int(fps)}]", (20 , frame.shape[0] - 20), font, font_scale, (0, 255, 0), font_thickness, lineType=cv2.LINE_AA)
    
    # define detection lists
    detections = np.empty((0, 5))
    
    # get detection results
    for r in results:
        # get coco classname
        class_name = r.names   
        boxes = r.boxes
        
        # get all bbox
        for bbox in boxes:
            #  get bbox coordinate
            x1, y1, x2, y2 = bbox.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # get current classname
            cls = int(bbox.cls[0])
            current_class = class_name[cls]
            
            # get confidents
            conf = math.ceil(bbox.conf[0] * 100) / 100
            
            if current_class in ["car"] and conf > 0.5:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

        # draw couter line
        cv2.line(frame, (line_couter[0], line_couter[1]), (line_couter[2], line_couter[3]), (0, 0, 255), 3)

        # SORT resutls 
        resultsTracker = tracker.update(detections)
        
        for result in resultsTracker:
            x1, y1, x2, y2, Id = result
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # get width, height
            w, h = x2 - x1, y2 - y1
            
            # mark center object
            cx, cy = x1 + w // 2, y1 + h // 2
            
            # count tracker
            if  line_couter[0] - 10 < cx < line_couter[2] + 10 and line_couter[1] - 3 < cy < line_couter[1] + 3:
                # check id not exist 
                if totalCount.count(Id) == 0:
                    totalCount.append(Id)
                    # holding counter line
                    cv2.line(frame, (line_couter[0], line_couter[1]), (line_couter[2], line_couter[3]), (0, 255, 0), 3)
    
            # confidents and tracker id
            current_id = int(Id)
            text = f"ID: {current_id}"
            
            # draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (colors[current_id % len(colors)]), rect_thickness)
            
            # calculate background label
            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
            x2, y2 = x1 + text_width, y1 - text_height - baseline
            
            # draw center point
            cv2.circle(frame, (cx, cy), 3, (colors[current_id % len(colors)]), cv2.FILLED)
            
            # draw label and background 
            cv2.rectangle(frame, (max(0, x1), max(35, y1)), (x2, y2), (colors[current_id % len(colors)]), cv2.FILLED)
            cv2.putText(frame, text, (max(0, x1), max(35, y1)), font, font_scale, white_text, font_thickness, lineType=cv2.LINE_AA)
        
    # draw counter text
    cv2.putText(frame, f"{len(totalCount)}", (frame.shape[1] - 150, frame.shape[0] - 50), font, (font_scale + 2), (255, 0, 255), (font_thickness + 2), lineType=cv2.LINE_AA)
    
    cv2.imshow("Frame", frame)
    cap_out.write(frame)
    
    key = cv2.waitKey(1)
    if key == 27: 
        break
    
cap.release()
cap_out.release()
cv2.destroyAllWindows()