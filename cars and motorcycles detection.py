import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from tracker import*
# import cvzone

model = YOLO("yolov8s")

cap = cv2.VideoCapture('4_5892973916439187503.mp4')

car_pass = {}
motorcycle_pass = {}
car_count = 1
motorcycle_count = 0

tracker=Tracker()

while True:
	ret, frame = cap.read()
	if ret:
		x1b = 45
		y1b = 540
		x2b = 1021

		cv2.line(frame,(x1b,y1b),(x2b,y1b),(0,255,0),2)

		results = model.predict(source=frame)

		list1 = []
		list2 = []

		for result in results:
			for (obj_xyxy, obj_cls) in zip(result.boxes.xyxy, result.boxes.cls):
				obj_cls = int(obj_cls)

				if obj_cls == 2  or obj_cls == 3:
					x1 = int(obj_xyxy[0].item())
					y1 = int(obj_xyxy[1].item())
					x2 = int(obj_xyxy[2].item())
					y2 = int(obj_xyxy[3].item())
					list1.append([x1,y1,x2,y2,obj_cls])
		
		bbox_id1=tracker.update(list1)
		for bbox in bbox_id1:
			x3,y3,x4,y4,ob,id=bbox
			cx=int(x3+x4)//2
			cy=int(y3+y4)//2
			
			if ob == 2:	
				# cv2.circle(frame,(cx,cy),4,(255,0,255),-1)
				if y1b<y4+15  and  y1b>y4-15:
					cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
					# cvzone.putTextRect(frame, f"{id}", (x3,y3),1,2)
					car_pass[id] = (cx,cy)
					car_count = len(car_pass) + 1

			elif ob == 3:	
				# cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
				if y1b<y4+30  and  y1b>y4-30:
					cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
					# cvzone.putTextRect(frame, f"{id}", (x3,y3),1,2)
					motorcycle_pass[id] = (cx,cy)
					motorcycle_count = len(motorcycle_pass)


		cv2.putText(frame, 'Cars: {}'.format(car_count), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.putText(frame, 'Motorcycles: {}'.format(motorcycle_count), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
		cv2.imshow("Webcam", frame)
		
		q = cv2.waitKey(1)
		if q == ord('q'):
			break
	else:
		break

cap.release()
cv2.destroyAllWindows()	