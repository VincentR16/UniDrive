
#TODO bisogna sicuramente adderstralo meglio, ma funzione gia molto bene, serve capire se si deve addestrare in un mondo reeale o nella pista della macchinina.
import cv2
from ultralytics import YOLO

model = YOLO('yolov8n.pt',verbose=False)
cap = cv2.VideoCapture('/Users/vincenzo/Desktop/VM/Progetto Fvab/CrossingStreet.mp4') #! inserire il path del video da analizzare

while cap.isOpened():
    ret, frame= cap.read()
    if not ret:
        break

    result = model(frame, device='mps', classes=[0],verbose=False,conf=0.51) 
    annoted_frame = result[0].plot()
    cv2.imshow('Obstacle Detection', annoted_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()