

#-----let's define a function to extract the all features in a folder. This function receives the folder, number of the features, a model to extract the data and the saved file name then it tries to extrac data from all 4 consecutive frames. Finally, it concatenates them in one matix and saves the matrix 
def data_ext(folder,name, number_of_feature,downsample_factor1,downsample_factor2,data_extractor):
    
    mean = 0
    var = 1
    sigma = var**0.5
    HF=np.zeros([number_of_feature,0])
    
    labels=[]
    datanum=0
    f = open(folder+name+'.txt', "r+")
    annot = f.readlines()                                                                 # in file haye text har pooshe ro mikhone
    f.close()
    print(name)  
    videos = glob.glob(folder+"/*.avi")
    videos.sort(key=natural_keys)
    i=0
    for j,vid in (enumerate(videos)):
            # if ((i)%4!=0):          #agar dade not fall hast #agar shomareye frame mazrabi az downsampling facter nist                        
            #     # print(f_num," :not extracted",end="   ")
            #     continue

            # print("\n")
            # print(vid, annot[i])
            
            start=int(annot[j].split()[0])
            end=int(annot[j].split()[1])
            f_all=int(annot[j].split()[2])

            for rest in range(0,1):
                cap = cv2.VideoCapture(vid)
                num_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if f_all!=num_frame:
                    print("error, num frames", num_frame)
                    print(vid, annot[j])
                
                f_num=0
                while(True):
                        

                    ret, frame = cap.read()
                    if ret==True and f_num<(num_frame-4) and i==0:
                        # f_num+=1
                        i=1
                        # print(f_num," :not extracted",end="   ")
                        if f_num-int(f_num/4)*4==rest:                            
                          # print(f_num," : extracted",end="   ")
                          p1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                          ret, frame = cap.read()
                          f_num+=1
                          p2 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                          ret, frame = cap.read()
                          f_num+=1
                          p3 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                          ret, frame = cap.read()
                          f_num+=1
                          p4 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                          p4=np.array(p4)
                          p3=np.array(p3)
                          p2=np.array(p2)
                          p1=np.array(p1)
                          FEAT=data_extractor(p1,p2,p3,p4)
                          maximum=np.amax(FEAT)
                          # FEAT=FEAT/maximum
                          row= FEAT.shape
                          if start==0 and end==0:
                            label=-1
                          elif f_num>=start and f_num<=end:
                            label=1

                          else:
                            label=-1                                                           
                          
                          HF=np.column_stack((HF,FEAT))
                          labels.append(label)
                          # print(f_num,end =" ")
                          # if f_num>=start and f_num<=end:
                    
                          #     for i in range(0,6):

                          #         gauss = np.random.normal(mean,sigma,(row))

                          #         FEAT_cpy=FEAT+(0.0001*gauss)
                          #         HF=np.column_stack((HF,FEAT_cpy))
                          #         # print(np.amax(FEAT),np.amax(FEAT_cpy),end="  ")
                          #         labels.append(1)

                    else:
                            break

    HF=np.array(HF)
    labels=np.array(labels)
    labels=labels.transpose()
    HF=HF.transpose()

    return HF,labels
