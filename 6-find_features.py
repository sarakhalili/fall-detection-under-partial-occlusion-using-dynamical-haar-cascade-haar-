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
# path_to_save=r"F:\fall detection\codes\2-Fall-Detection_Dynamical_haar_Cascade-master\new_samples_clue.txt"

# svc=SVC(probability=True, kernel='linear')
# data=np.loadtxt(path_to_save, delimiter=',')
# n_number=100
# model=joblib.load(r"C:\Users\elec2\Downloads\fall_model100.sav")
# feature_importances=model.feature_importances_
# for i in range(0,int(feature_importances.shape[0])):
# 	if feature_importances[i]>0:
# 		print(int(feature_importances.shape[0]))
def find_feature(data):
	feat_para_0=data[1]
	feat_para_1=data[2]
	feat_para_2=data[3]
	feat_para_3=data[4]
	feat_para_4=data[5]
	feat_para_5=data[6]
	if feat_para_5==1:
		return("r(delta10) -r(U1)\n")
	elif feat_para_5==2:
		return ("r(delta10) -r(D1)\n")
	elif feat_para_5==3:
		return ("r(delta10) -r(R1)\n")
	elif feat_para_5==4:
		return ("r(delta10) -r(L1)\n")
	elif feat_para_5==5:
		return ("r(delta20) -r(U2)\n")
	elif feat_para_5==6:
		return ("r(delta20) -r(D2)\n")
	elif feat_para_5==7:
		return ("r(delta20) -r(R2)\n")
	elif feat_para_5==8:
		return ("r(delta20) -r(L2)\n")
	elif feat_para_5==9:
		return ("r(delta30) -r(U3)\n")
	elif feat_para_5==10:
		return ("r(delta30) -r(D3)\n")
	elif feat_para_5==11:
		return ("r(delta30) -r(R3)\n")
	elif feat_para_5==12:
		return ("r(delta30) -r(L3)\n")
	elif feat_para_5==13:
		return ("r(delta21) -r(U2)\n")
	elif feat_para_5==14:
		return ("r(delta21) -r(D2)\n")
	elif feat_para_5==15:
		return ("r(delta21) -r(R2)\n")
	elif feat_para_5==16:
		return ("r(delta21) -r(L2)\n")
	elif feat_para_5==17:
		return ("r(delta31) -r(U3)\n")
	elif feat_para_5==18:
		return ("r(delta31) -r(D3)\n")
	elif feat_para_5==19:
		return ("r(delta31) -r(R3)\n")
	elif feat_para_5==20:
		return ("r(delta31) -r(L3)\n")
	elif feat_para_5==21:
		return ("r(delta32) -r(U3)\n")
	elif feat_para_5==22:
		return ("r(delta32) -r(D3)\n")
	elif feat_para_5==23:
		return ("r(delta32) -r(R3)\n")
	elif feat_para_5==24:
		return ("r(delta32) -r(L3)\n")
	elif feat_para_5==49:
		return ("r(dU10)\n")
	elif feat_para_5==50:
		return ("r(dU20)\n")
	elif feat_para_5==51:
		return ("r(dU30)\n")
	elif feat_para_5==52:
		return ("r(dU21)\n")
	elif feat_para_5==53:
		return ("r(dU31)\n")
	elif feat_para_5==54:
		return ("r(dU32)\n")
	elif feat_para_5==55:
		return ("r(dD10)\n")
	elif feat_para_5==56:
		return ("r(dD20)\n")
	elif feat_para_5==57:
		return ("r(dD30)\n")
	elif feat_para_5==58:
		return ("r(dD21)\n")
	elif feat_para_5==59:
		return ("r(dD31)\n")
	elif feat_para_5==60:
		return ("r(dD32)\n")
	elif feat_para_5==61:
		return ("r(dR10)\n")
	elif feat_para_5==62:
		return ("r(dR20)\n")
	elif feat_para_5==63:
		return ("r(dR30)\n")
	elif feat_para_5==64:
		return ("r(dR21)\n")
	elif feat_para_5==65:
		return ("r(dR31)\n")
	elif feat_para_5==66:
		return ("r(dR32)\n")
	elif feat_para_5==67:
		return ("r(dL10)\n")
	elif feat_para_5==68:
		return ("r(dL20)\n")
	elif feat_para_5==69:
		return ("r(dL30)\n")
	elif feat_para_5==70:
		return ("r(dL21)\n")
	elif feat_para_5==71:
		return ("r(dL31)\n")
	elif feat_para_5==72:
		return ("r(dL32)\n")
	else:
		if feat_para_5==25:
			feat_input="dU10"
		elif feat_para_5==26:
			feat_input="dU20"
		elif feat_para_5==27:
			feat_input="dU30"
		elif feat_para_5==28:
			feat_input="dU21"
		elif feat_para_5==29:
			feat_input="dU31"
		elif feat_para_5==30:
			feat_input="dU32"
		elif feat_para_5==31:
			feat_input="dD10"
		elif feat_para_5==32:
			feat_input="dD20"
		elif feat_para_5==33:
			feat_input="dD30"
		elif feat_para_5==34:
			feat_input="dD21"
		elif feat_para_5==35:
			feat_input="dD31"
		elif feat_para_5==36:
			feat_input="dD32"
		elif feat_para_5==37:
			feat_input="dR10"
		elif feat_para_5==38:
			feat_input="dR20"
		elif feat_para_5==39:
			feat_input="dR30"
		elif feat_para_5==40:
			feat_input="dR21"
		elif feat_para_5==41:
			feat_input="dR31"
		elif feat_para_5==42:
			feat_input="dR32"
		elif feat_para_5==43:
			feat_input="dL10"
		elif feat_para_5==44:
			feat_input="dL20"
		elif feat_para_5==45:
			feat_input="dL30"
		elif feat_para_5==46:
			feat_input="dL21"
		elif feat_para_5==47:
			feat_input="dL31"
		elif feat_para_5==48:
			feat_input="dL32"
		elif feat_para_5==73:
			feat_input="a"
		elif feat_para_5==74:
			feat_input="b"
		elif feat_para_5==75:
			feat_input="c"
		elif feat_para_5==76:
			feat_input="d"
		if feat_para_0==66:
			return("featmul({},{},{},{})\n".format(feat_input,int(feat_para_1),int(feat_para_2),int(feat_para_3)))
		elif feat_para_0==12:
			return("feat12({},{},{},{},{})\n".format(feat_input,int(feat_para_1),int(feat_para_2),int(feat_para_3),int(feat_para_4)))
		elif feat_para_0==21:
			return("feat21({},{},{},{},{})\n".format(feat_input,int(feat_para_1),int(feat_para_2),int(feat_para_3),int(feat_para_4)))
		elif feat_para_0==13:
			return("feat13({},{},{},{},{})\n".format(feat_input,int(feat_para_1),int(feat_para_2),int(feat_para_3),int(feat_para_4)))
		elif feat_para_0==22:
			return("feat22({},{},{},{},{})\n".format(feat_input,int(feat_para_1),int(feat_para_2),int(feat_para_3),int(feat_para_4)))

