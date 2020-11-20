import cv2
import numpy as np
import pandas as pd

face_cascasde=cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
eye_cascade=cv2.CascadeClassifier('train/third-party/frontalEyes35x16.xml')
nose_cascade=cv2.CascadeClassifier('train/third-party/Nose18x15.xml')
glasses=cv2.imread('train/cat_eye.png',cv2.IMREAD_UNCHANGED)
mustache=cv2.imread('train/mustache.png',cv2.IMREAD_UNCHANGED)
print(glasses.shape)
print(mustache.shape)
cap=cv2.VideoCapture(0)
while True:
	ret,frame=cap.read()
	if ret==False:
		print('Can not open the webcam')
		continue

	frame1=cv2.cvtColor(frame,cv2.COLOR_RGB2RGBA)
	faces=face_cascasde.detectMultiScale(frame1,1.3,5)
	for x,y,w,h in faces:
		cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
		face_section=frame1[y:y+h,x:x+w]
		eyes=eye_cascade.detectMultiScale(face_section,1.4)
		nose=nose_cascade.detectMultiScale(face_section,1.6)
		print(nose)
		print(eyes)

		for [ex,ey,ew,eh] in eyes:
			#cv2.rectangle(face_section,(ex,ey),(ex+eh,ey+ew),(255,0,0),2)	
			glasses=cv2.resize(glasses,(eh,ew))
			row,col,channel=glasses.shape
			for i in range(row):
					for j in range(col):
						if glasses[i][j][3]!=0:
							face_section[ey+i][ex+j]=glasses[i][j]
		for nx,ny,nw,nh in nose:
			#cv2.rectangle(face_section,(nx,ny),(nx+nh,ny+nw),(255,0,0),2)
			mustache=cv2.resize(mustache,(nh,nw))
			row1,col1,channel=mustache.shape
			for a in range(row1):
				for b in range(col1):
					if mustache[a][b][3]!=0:
						face_section[ny+a+12][nx+b]=mustache[a][b]


	key_pressed=cv2.waitKey(2) & 0xFF
	if key_pressed==ord('q'):
 		break
	cv2.imshow('video frames',frame1)
cap.release()
cv2.destroyAllWindows()

