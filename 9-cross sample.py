from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA as sklearnPCA

# --------------extract features--------------------
# -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
data_folderh="/content/drive/MyDrive/full_frame/original/"
datasets=[['Home_01','Home_02','Lecture_room','Office','Coffee_room_01','ur_fall','ur_adl']]

for folder in datasets:
    for i,ds in (enumerate(folder)):
        x,y=data_ext(data_folderh+ds+'/',ds,n_feature,1,1,fall_model)

        if i==0:
          all_feat_n=x
          labels_n=y
        else:
          all_feat_n=np.concatenate((all_feat_n,x),axis=0)
          labels_n=np.concatenate((labels_n,y),axis=0)        

data_folderh="/content/drive/MyDrive/full_frame/openpose/"
datasets=[['Home_01','Home_02','Lecture_room','Office','Coffee_room_01','ur_fall','ur_adl']]

for folder in datasets:
    for i,ds in (enumerate(folder)):
        x,y=data_ext(data_folderh+ds+'/',ds,n_feature,1,1,fall_model)

        if i==0:
          all_feat_o=x
          labels_o=y
        else:
          all_feat_o=np.concatenate((all_feat_o,x),axis=0)
          labels_o=np.concatenate((labels_o,y),axis=0)        

#-------------test and train data -----------------------------------
# trained_X=all_feat_n; trained_Y=labels_n
# X1, Xvalid, Y1, Yvalid =  train_test_split(all_feat_n, labels_n, test_size=0.2)
X=np.concatenate((all_feat_n,all_feat_o),axis=0); Y=np.concatenate((labels_n,labels_o),axis=0)
trained_X, trained_X_test, trained_Y, trained_y_test =  train_test_split(X, Y, test_size=0.2)
trained_X, Xvalid, trained_Y, Yvalid =  train_test_split(trained_X, trained_Y, test_size=0.15)
print("Xtrain is:",np.shape(trained_X),"          Xtest is:",np.shape(trained_X_test))   
print("Ytrain is:",np.shape(trained_Y),"          Ytest is:",np.shape(trained_y_test)) 
# del trained_all_features, trained_labels
# print(trained_all_features.shape)
# -------------------Adaboost-----------------------------------------
# abc = AdaBoostClassifier(n_estimators=n_feature,learning_rate=1)
# trained_model = abc.fit(trained_X, trained_Y)
# trained_y_pred = trained_model.predict(trained_X_test)
# # #-----------------find accuracy----------------------------------------

# # print("AdaBoost 2's Accuracy:",metrics.accuracy_score(trained_y_test, trained_y_pred))
# print("classification_report:\n",classification_report(trained_y_test, trained_y_pred, labels=[-1,1]))
# print("confusion_matrix:\n",confusion_matrix(trained_y_test, trained_y_pred))

# -------------------cross validation--------------------------------------------
#-----------------------------------------------------------------------
C=[1,10,100]
Degree=[2,3,4,5]
score=[]
for c in C:
  for d in Degree:
      svclassifier1 = SVC(kernel='poly',degree=d,C=c,class_weight='balanced',)
      svclassifier1.fit(trained_X,trained_Y)
      trained_y_pred = svclassifier1.predict(Xvalid)
      report=confusion_matrix(Yvalid, trained_y_pred, labels=[-1,1])
      print(report,c,d)
      s1=report[0,0]/(report[0,0]+report[0,1])
      s2=report[1,1]/(report[1,0]+report[1,1])
      s=(s1+(1*s2))
      score.append([s,c,d])
score=np.array(score)
score=score[score[:, 0].argsort()]
print(score)
# svclassifier1 = SVC(kernel='poly',degree=score[-1,2],C=score[-1,1],class_weight='balanced',)
# svclassifier1.fit(trained_X,trained_Y)
svclassifier1 = SVC(kernel='poly',degree=4,C=1,class_weight='balanced',)
svclassifier1.fit(trained_X,trained_Y)
# trained_y_pred = svclassifier1.predict(trained_X_test)
# print("confusion_matrix:\n",confusion_matrix(trained_y_test, trained_y_pred))
# print("classification_report:\n",classification_report(trained_y_test, trained_y_pred, labels=[-1,1]))
# abc = AdaBoostClassifier(n_estimators=n_feature,learning_rate=1)
# print("val_score:",np.mean(cross_val_score(abc, all_feat, labels, cv=10)))

# pca=sklearnPCA(n_components=2)
# X = np.concatenate([trained_X, trained_X_test])
# pca.fit(X)
# Xtrain = pca.transform(trained_X)
# Xtest = pca.transform(trained_X_test)        
# del X; del pca
# trained_Y
# plt.scatter(Xtrain[:,0], Xtrain[:,1], c =trained_Y)
# plt.show()
# plt.scatter(Xtest[:,0], Xtest[:,1], c =trained_y_test) 
# plt.show()
# plt.scatter(Xtest[:,0], Xtest[:,1], c =trained_y_pred) 
# plt.show()
#------------------------------------------------------------------------------------------------------
