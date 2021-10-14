# this code tries to find which features are selected by Adaboost, the haarfeatures_with_clue tries to label all feature then this code findes the selected features using the labels (feat_paras) and returns the string which is used to extract the features. instead of labeling all faetures in each try, one matrix named new_samples_clue.py is saved, if you apply any change in the feature extraction scenario, a new matrix should be saved
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import numpy as np
import time
import sklearn.externals
import glob
import joblib

path_to_save="/content/drive/My Drive/new_samples_clue.txt"

svc=SVC(probability=True, kernel='linear')
data=np.loadtxt(path_to_save, delimiter=',')
# n_number=100
model=joblib.load("/content/drive/MyDrive/model/fall_model_v5.sav")
#------------------find which feature have been chosen by Adaboost-------
feature_importances=model.feature_importances_
# print(feature_importances.shape[0])

# feature_importances=model.feature_importances_
# for i in range(0,int(feature_importances.shape[0])):
	# if feature_importances[i]>0:
		# print(int(feature_importances.shape[0]))
#-----------------write a new code-------------
	#----------first the features should be labled (if your data extraction scenario is not same as your previous scenario, you should first save a new matrix named new_samples_clue.py using features function in haarfeature_with_clue.py. Be careful to change the data extraction scenario in this folder too.
#first method
# -*- coding: utf-8 -*-

file = open("/content/drive/MyDrive/final_model.py","w") 

# file.write("from haarfeature import * \n") 
file.write("import numpy as np \n") 
file.write("def fall_model(a,b,c,d):\n")
file.write("    U1=move_frame(b,0)\n")
file.write("    D1=move_frame(b,1)\n")
file.write("    R1=move_frame(b,2)\n")
file.write("    L1=move_frame(b,3)\n")
file.write("    U2=move_frame(c,0)\n")
file.write("    D2=move_frame(c,1)\n")
file.write("    R2=move_frame(c,2)\n")
file.write("    L2=move_frame(c,3)\n")
file.write("    U3=move_frame(d,0)\n")
file.write("    D3=move_frame(d,1)\n")
file.write("    R3=move_frame(d,2)\n")
file.write("    L3=move_frame(d,3)\n")
file.write("    dU10=np.bitwise_xor(U1,a)\n")
file.write("    dD10=np.bitwise_xor(D1,a)\n")
file.write("    dR10=np.bitwise_xor(R1,a)\n")
file.write("    dL10=np.bitwise_xor(L1,a)\n")
file.write("    delta10=np.bitwise_xor(b,a)\n")
file.write("    dU20=np.bitwise_xor(U2,a)\n")
file.write("    dD20=np.bitwise_xor(D2,a)\n")
file.write("    dR20=np.bitwise_xor(R2,a)\n")
file.write("    dL20=np.bitwise_xor(L2,a)\n")
file.write("    delta20=np.bitwise_xor(c,a)\n")
file.write("    dU30=np.bitwise_xor(U3,a)\n")
file.write("    dD30=np.bitwise_xor(D3,a)\n")
file.write("    dR30=np.bitwise_xor(R3,a)\n")
file.write("    dL30=np.bitwise_xor(L3,a)\n")
file.write("    delta30=np.bitwise_xor(d,a)\n")
file.write("    dU21=np.bitwise_xor(U2,b)\n")
file.write("    dD21=np.bitwise_xor(D2,b)\n")
file.write("    dR21=np.bitwise_xor(R2,b)\n")
file.write("    dL21=np.bitwise_xor(L2,b)\n")
file.write("    delta21=np.bitwise_xor(c,b)\n")
file.write("    dU31=np.bitwise_xor(U3,b)\n")
file.write("    dD31=np.bitwise_xor(D3,b)\n")
file.write("    dR31=np.bitwise_xor(R3,b)\n")
file.write("    dL31=np.bitwise_xor(L3,b)\n")
file.write("    delta31=np.bitwise_xor(d,b)\n")
file.write("    dU32=np.bitwise_xor(U3,c)\n")
file.write("    dD32=np.bitwise_xor(D3,c)\n")
file.write("    dR32=np.bitwise_xor(R3,c)\n")
file.write("    dL32=np.bitwise_xor(L3,c)\n")
file.write("    delta32=np.bitwise_xor(d,c)\n")
file.write("    ###########################################################################\n")

       
feat_count=1
for i in range(0,int(feature_importances.shape[0])):
 	if feature_importances[i]>0:
        #  print(feature_importances[i],i)
         new_line=find_feature(data[i,:])
         feat_num='    '+'feat_{}'.format(feat_count)
         file.write(feat_num+"="+new_line)
         feat_count+=1

file.write("    ###########################################################################\n")               
file.write("    feat=[")         
# file.write('    '+"feat_count={}".format(feat_count)+"\n")
for i in range(1,feat_count-1):
        feat_num='feat_{}'.format(i)
        file.write(feat_num+",")
file.write('feat_{}'.format(feat_count-1)+"]\n")
        
file.write("    feat=np.array(feat)\n")
file.write("    feat=feat.transpose()\n")
file.write("    return feat \n")
file.close()
n_feature=feat_count-1
