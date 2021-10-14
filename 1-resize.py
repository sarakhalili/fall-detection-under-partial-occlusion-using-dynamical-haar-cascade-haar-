# this code tries to convert the fall and not-fall videos to the desired form which is 20*15-sized-frame videos, then it saves the new movies in another folder

import numpy as np 
import cv2
import os; import glob
import re
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def resize(first_folder,target_folder,gam,name):
    f = open(first_folder+name+'.txt', "r+")
    annot = f.readlines()                                                                 
    f.close()

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    
    vid_num=0
    
    videos = glob.glob(first_folder+"/*.avi")
    videos.sort(key=natural_keys)
    mmm=len(videos)
    uuu=int(0.8*mmm)
    for j,vid in (enumerate(videos)):
          
        start=int(annot[j].split()[0])
        end=int(annot[j].split()[1])
        f_all=int(annot[j].split()[2])
        vid_num+=1           
        cap = cv2.VideoCapture(vid)
        num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        framerate = cap.get(cv2.CAP_PROP_FPS)
        #  print(num_frame)
        if j<uuu:
          folder='train'
        else:
          folder='test'
        out = cv2.VideoWriter('{}/video_{}.avi'.format(target_folder+folder,vid_num+gam),fourcc, framerate, (15,20),True)
        f2 = open(target_folder+folder+'/'+folder+".txt", "a")
        f2.write('{} {} {}\n'.format(start,end ,f_all))
        f2.close()
        for i in range(0, num_frame+1):
              
              ret, frame=cap.read()
              if ret==True:
                 if i==0:
                      r,c,_ = frame.shape # here the biggest square in the middle of frame is cropped
                      d = (c-r)//2
                      sc = int(d+(r/6))
                      ec =int(r-(r/6)+d)
                 # print(frame.shape,'  ', sc,ec)
                 resizedfram = cv2.resize(frame[:, sc:ec], (15,20), interpolation = cv2.INTER_AREA)
                 out.write(resizedfram)
              else:
                  out.release()
                  break
#-------------------------resize data set with the defined function named resize which receives the lower part files path and resize the movies in the directory to 15*20 movies in the new directory path which also should be received as one of the inputs------------------------------
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/Coffee_room_01",r"/content/drive/MyDrive/yolo/openpose/Coffee_room_01",100)
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/Home_01",r"/content/drive/MyDrive/yolo/openpose/Home_01",200)
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/Home_02",r"/content/drive/MyDrive/yolo/openpose/Home_02",300)
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/Lecture_room",r"/content/drive/MyDrive/yolo/openpose/Lecture_room",400)
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/Office",r"/content/drive/MyDrive/yolo/openpose/Office",500)
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/Office2",r"/content/drive/MyDrive/yolo/openpose/Office2",600)
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/ur_fall",r"/content/drive/MyDrive/yolo/openpose/ur_fall",700)
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/ur_adl",r"/content/drive/MyDrive/yolo/openpose/ur_adl",800)      
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/mine",r"/content/drive/MyDrive/yolo/openpose/mine")
# resize(r"/content/drive/MyDrive/openpose/circle/yolo/mine_fall",r"/content/drive/MyDrive/yolo/openpose/mine_fall")     
tttype='openpose'        
!mkdir "/content/drive/MyDrive/full_frame/cross_subject/openpose/" 
!mkdir "/content/drive/MyDrive/full_frame/cross_subject/openpose/train/"    
!mkdir "/content/drive/MyDrive/full_frame/cross_subject/openpose/test/"    

dest="/content/drive/MyDrive/full_frame/cross_subject/openpose/"
first='/content/drive/MyDrive/full_frame/openpose/'
datasets=[['Home_01','Home_02','Office2','Lecture_room','Office','Coffee_room_01','ur_fall','ur_adl']]

for folder in datasets:
    for i,ds in (enumerate(folder)):
        resize(first+ds+'/',dest,i*100,ds)