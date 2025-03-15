import numpy as np
import imutils
import time
import cv2
from imutils.video import VideoStream
import tkinter as tk
from tkinter import messagebox

LABELS = open("coco.names").read().strip().split("\n")


print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet("person.cfg", "person.weights")


np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

ln = net.getLayerNames()
ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]

vs = VideoStream(src=0).start()
(W, H) = (None, None)

popup_shown = 0
last_popup_time = 0
popup_duration = 5  

def show_popup():
    global popup_shown, last_popup_time
    popup_shown += 1
    last_popup_time = time.time()
    
    root = tk.Tk()
    root.withdraw()  #
    messagebox.showinfo("Alert", "Cell phone detected!")
    root.after(popup_duration * 1000, root.destroy)  
    root.mainloop()

while True:
    
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    if W is None or H is None:
        (H, W) = frame.shape[:2]
 
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (224, 224),
        swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()
 
    boxes = []
    confidences = []
    classIDs = []
    centers = []

    for output in layerOutputs:
       
        for detection in output:
        
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
 
            if confidence > 0.4 and classID==67:  
            
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
 
            
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
 
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
                centers.append((centerX, centerY))

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    if len(idxs) > 0:
        current_time = time.time()
       
        if popup_shown < 3 and (current_time - last_popup_time > popup_duration):
            show_popup()
        
        
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
           
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                confidences[i])
            cv2.putText(frame, text, (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    

    cv2.imshow("Image", frame)
    
    key = cv2.waitKey(1) & 0xFF
   
    if key == ord("q"):
        break
    
    if popup_shown >= 3:
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo("Alert", "Exam stopped!")
        root.after(3000, lambda: root.destroy())  
        root.mainloop()
        break
print("[INFO] cleaning up...")
vs.stop()
cv2.destroyAllWindows()
