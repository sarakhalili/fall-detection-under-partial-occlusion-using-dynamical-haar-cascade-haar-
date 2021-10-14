# import needed libraries
import cv2.cv2 as cv2
from haarfeature_with_clue import features
import numpy as np

path_to_video=r"C:\Users\elec2\Desktop\fall\dataset\fall\vid_1_1.avi"
path_to_save=r"C:\Users\elec2\Desktop\fall\2-Fall-Detection_haar\new_samples_clue.txt"
cap = cv2.VideoCapture(path_to_video)
ret, frame1 = cap.read()
frame1=cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
ret, frame2 = cap.read()
frame2=cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
ret, frame3 = cap.read()
frame3=cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
ret, frame4 = cap.read()
frame4=cv2.cvtColor(frame4, cv2.COLOR_BGR2GRAY)
resizedfram1=cv2.resize(frame1,(15,20),interpolation=cv2.INTER_AREA)
resizedfram2=cv2.resize(frame2,(15,20),interpolation=cv2.INTER_AREA)
resizedfram3=cv2.resize(frame3,(15,20),interpolation=cv2.INTER_AREA)
resizedfram4=cv2.resize(frame4,(15,20),interpolation=cv2.INTER_AREA)

feat=features(resizedfram1,resizedfram2,resizedfram3,resizedfram4)
np.savetxt(path_to_save, feat, delimiter=',')
