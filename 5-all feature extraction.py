#------------------import libraries
import cv2
import numpy as np
import glob
from tqdm import tqdm
import re
from os import path

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):

    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

#-----let's define a function to extract the all features in a folder. This function receives the folder, number of the features, a model to extract the data and the saved file name then it tries to extrac data from all 4 consecutive frames. Finally, it concatenates them in one matix and saves the matrix 
def data_ext(folder,name, number_of_feature,data_extractor, saved_x,save_y):    
    f = open(folder+name+'.txt', "r+")
    annot = f.readlines()                                                                 # in file haye text har pooshe ro mikhone
    f.close()
    print(name)  
    videos = glob.glob(folder+"/*.avi")
    videos.sort(key=natural_keys)
    for i,vid in (enumerate(videos)):
        print("\n")
        print(vid, annot[i])
        HF=np.zeros([number_of_feature,0])
        labels=[]
        start=int(annot[i].split()[0])
        end=int(annot[i].split()[1])
        f_all=int(annot[i].split()[2])
        
        # print(start,end,f_all)
        for rest in range(0,1):
            cap = cv2.VideoCapture(vid)
            num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if f_all!=num_frame:
                print("error, num frames", num_frame,f_all)
            i=0
            f_num=0
            while(True):
                # Capture frame-by-frame
                ret, frame = cap.read()
                if ret==True:
                    f_num+=1
                    if f_num-int(f_num/4)*4==rest:
                        i+=1
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        if i==1:
                            p1=np.array(frame)
                        elif i==2:
                            p2=np.array(p1)
                            p1=np.array(frame)
                        elif i==3:
                            p3=np.array(p2)
                            p2=np.array(p1)
                            p1=np.array(frame)
                        else:
                            p4=np.array(p3)
                            p3=np.array(p2)
                            p2=np.array(p1)
                            p1=np.array(frame)
                            FEAT=data_extractor(p1,p2,p3,p4)
                            if start==0 and end==0:
                              label=-1
                            elif f_num>=start and f_num<=end:
                              label=1

                            else:
                              label=-1     
                                                             
                            
                            HF=np.column_stack((HF,FEAT))
                            labels.append(label)
                            print(f_num,end =" ")

                else:
                        break                 
        HF=HF.transpose()
        labels=np.array(labels)
        labels=labels.transpose()
        if path.exists(saved_x):
            features=(np.load(saved_x))
            labels_all=(np.load(saved_y))
            features=np.concatenate((features,HF),axis=0)
            labels_all=np.concatenate((labels_all,labels),axis=0)
            np.save(saved_x,features)
            np.save(saved_y,labels_all)
        else:
            features=HF
            labels_all=labels
            np.save(saved_x,HF)
            np.save(saved_y,labels)
        print('video_{} Done'.format(i))
        print('this video shape: {} --- total shape:{}'.format(np.shape(HF),np.shape(features)))

 #------------------import libraries
import numpy as np
import glob
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
# Import train_test_split function
from sklearn.model_selection import train_test_split  
from sklearn import metrics
import numpy as np
import cv2.cv2 as cv2
import time
import sklearn.externals
import joblib
# --------------(!!!! time consuming !!!) extract all available features (if the features have extracted previously then comment first two lines and uncomment the other two line (Be careful about the namess))--------------------

data_folder="/content/drive/MyDrive/allfeatures/"

name='all_features'
saved_x="/content/drive/My Drive/"+name+'_diff.npy'
saved_y="/content/drive/My Drive/"+name+'_diff_labels.npy'

data_ext(data_folder,name,700272,features,saved_x,saved_y)



#####_________________________________if you have extracted the features in advance, load your saved arrays-------------------------------------

features=(np.load(saved_x))
labels=(np.load(saved_y))
# print(features)
print(np.shape(features),np.shape(labels))
print("loading done")

#-------------test and train data -----------------------------------
X, X_test, Y, y_test = train_test_split(features, labels, test_size=0.2)
del features, labels
 
# del X, X_test, Y, y_test
n_feature=300
#-------------------Adaboost-----------------------------------------
print("starting to train model")
abc = AdaBoostClassifier(n_estimators=n_feature,learning_rate=1)
print("fitting a model")
model = abc.fit(X, Y)
print("training done")
#-----------------find accuracy----------------------------------------
y_pred = model.predict(X_test)

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
from sklearn.metrics import classification_report, confusion_matrix
print("classification_report:\n",classification_report(y_test, y_pred, labels=[-1,1]))
print("confusion_matrix:\n",confusion_matrix(y_test, y_pred))
# # # ------------------save model-------------------------------------------
joblib.dump(model,"/content/drive/MyDrive/fall_model.sav") 

