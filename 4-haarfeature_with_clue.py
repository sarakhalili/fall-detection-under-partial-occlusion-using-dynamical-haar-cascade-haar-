import numpy as np
def Im(a,x,y):
    summed=0
    interested_area=a[0:y+1,0:x+1]
    summed=np.sum(interested_area)
    return summed
def Imrect(a,x1,y1,x2,y2):
    intrested_area=a[y1:y2+1,x1:x2+1]
    summed=np.sum(intrested_area)
    return summed
def feat21(a,x,y,z,k):
    bright=a[y:k+1,int((x+z)/2):z+1]
    dark=a[y:k+1,x:int((x+z)/2)+1]
    feature_21=np.sum(bright)-np.sum(dark)
    return feature_21
def feat12(a,x,y,z,k):
    bright=a[int((y+k)/2):k+1,x:z+1]
    dark=a[y:int((y+k)/2)+1,x:z+1]
    feature_12=np.sum(bright)-np.sum(dark)
    return feature_12
def feat13(a,x,y,z,k):
    bright1=a[y:k+1,x:int((2*x+z)/3)+1]
    bright2=a[y:k+1,int((x+2*z)/3):z+1]
    dark=a[y:k+1,int((2*x+z)/3):int((2*z+x)/3)+1]
    feat_13=np.sum(bright1)+np.sum(bright2)-2*np.sum(dark)
    return feat_13
def feat22(a,x,y,z,k):
    bright1=Imrect(a,x,y,int((x+z)/2),int((y+k)/2))
    bright2=Imrect(a,int((x+z)/2),int((y+k)/2),z,k)
    dark1=Imrect(a,x,int((y+k)/2),int((x+z)/2),k)
    dark2=Imrect(a,int((x+z)/2),y,z,int((y+k)/2))
    feat_22=bright1+bright2-dark1-dark2
    return feat_22
def featmul(a,x,y,l):
    z=x+2*l
    k=y+2*l
    bright1=Imrect(a,x-l,y,z-2*l,k)
    dark1=Imrect(a,x,y-l,z-l,k-l)
    bright2=Imrect(a,x+l,y+l,z,k+l)
    dark2=Imrect(a,x+2*l,y,z+l,k)
    feature_mul=bright1-dark1+bright2-dark2
    return feature_mul
def allfeat21(a):
    L=a.shape[1]
    H=a.shape[0]
    Out=np.zeros([0,6])
    for i in range(0,L):
        for j in range(0,H):
            for z in range(i+1,L):
                for k in range(j+2,H,2):
                    feat=feat21(a,i,j,z,k)
                    out_2=np.array([feat,21,i,j,z,k])
                    Out=np.row_stack((Out,out_2))
    return Out
def allfeat12(a):
    L=a.shape[1]
    H=a.shape[0]
    Out=np.zeros([0,6])
    for i in range(0,L):
        for j in range(0,H):
            for z in range(i+2,L,2):
                for k in range(j+1,H):
                    feat=feat12(a,i,j,z,k)
                    out_2=np.array([feat,12,i,j,z,k])
                    Out=np.row_stack((Out,out_2))

    return Out
def allfeat13(a):
    L=a.shape[1]
    H=a.shape[0]
    Out=np.zeros([0,6])
    for i in range(0,L):
        for j in range(0,H):
            for z in range(i+3,L,3):
                for k in range(j+1,H):
                    #print(i,j,z,k)
                    feat=feat13(a,i,j,z,k)
                    out_2=np.array([feat,13,i,j,z,k])
                    Out=np.row_stack((Out,out_2))

    return Out
def allfeat22(a):
    L=a.shape[1]
    H=a.shape[0]
    Out=np.zeros([0,6])
    for i in range(0,L):
        for j in range(0,H):
            for z in range(i+2,L,2):
                for k in range(j+2,H,2):
                    feat=feat22(a,i,j,z,k)
                    out_2=np.array([feat,22,i,j,z,k])
                    Out=np.row_stack((Out,out_2))
                   
    return Out
def allfeatmul(a):
    L=a.shape[1]
    H=a.shape[0]
    Out=np.zeros([0,6])
    for i in range(0,L):
        for j in range(0,H):
            dis1=i
            dis2=j
            dis3=L-i
            dis4=H-j
            maxl=min([dis1,dis2,int(dis3/3),int(dis4/3)])
            for l in range(1,maxl):
               feat=featmul(a,i,j,l)
               out_2=np.array([feat,66,i,j,l,0])
               Out=np.row_stack((Out,out_2))
             
    return Out
def r(a):
    L=a.shape[1]
    H=a.shape[0]
    feat=np.array([Im(a,L-1,H-1),99,99,99,99,99])
    return feat
def phi (a):
    feat2=allfeat12(a)
    feat3=allfeat21(a)
    feat1=allfeat13(a)
    feat4=allfeat22(a)
    feat5=allfeatmul(a)
    feat=np.row_stack((feat1,feat2,feat3,feat4,feat5))
    return feat
def move_frame (b,direction):
    if direction==0:   #up
        bnew=np.delete(b,0,0)
        bnew=np.row_stack((bnew,np.zeros([1,bnew.shape[1]],dtype=np.uint8)))
    elif direction==1: #down
        bnew=np.delete(b,b.shape[0]-1,0)
        bnew=np.row_stack((np.zeros([1,bnew.shape[1]],dtype=np.uint8),bnew))
    elif direction==2: #right
        bnew=np.delete(b,b.shape[1]-1,1)
        bnew=np.column_stack((np.zeros([bnew.shape[0],1],dtype=np.uint8),bnew))
    elif direction==3:
        bnew=np.delete(b,0,1)
        bnew=np.column_stack((bnew,np.zeros([bnew.shape[0],1],dtype=np.uint8)))
    return bnew
def features(a,b,c,d):
    U1=move_frame(b,0)
    D1=move_frame(b,1)
    R1=move_frame(b,2)
    L1=move_frame(b,3)
    U2=move_frame(c,0)
    D2=move_frame(c,1)
    R2=move_frame(c,2)
    L2=move_frame(c,3)
    U3=move_frame(d,0)
    D3=move_frame(d,1)
    R3=move_frame(d,2)
    L3=move_frame(d,3)
    dU10=np.bitwise_xor(U1,a)
    dD10=np.bitwise_xor(D1,a)
    dR10=np.bitwise_xor(R1,a)
    dL10=np.bitwise_xor(L1,a)
    delta10=np.bitwise_xor(b,a)
    dU20=np.bitwise_xor(U2,a)
    dD20=np.bitwise_xor(D2,a)
    dR20=np.bitwise_xor(R2,a)
    dL20=np.bitwise_xor(L2,a)
    delta20=np.bitwise_xor(c,a)
    dU30=np.bitwise_xor(U3,a)
    dD30=np.bitwise_xor(D3,a)
    dR30=np.bitwise_xor(R3,a)
    dL30=np.bitwise_xor(L3,a)
    delta30=np.bitwise_xor(d,a)
    dU21=np.bitwise_xor(U2,b)
    dD21=np.bitwise_xor(D2,b)
    dR21=np.bitwise_xor(R2,b)
    dL21=np.bitwise_xor(L2,b)
    delta21=np.bitwise_xor(c,b)
    dU31=np.bitwise_xor(U3,b)
    dD31=np.bitwise_xor(D3,b)
    dR31=np.bitwise_xor(R3,b)
    dL31=np.bitwise_xor(L3,b)
    delta31=np.bitwise_xor(d,b)
    dU32=np.bitwise_xor(U3,c)
    dD32=np.bitwise_xor(D3,c)
    dR32=np.bitwise_xor(R3,c)
    dL32=np.bitwise_xor(L3,c)
    delta32=np.bitwise_xor(d,c)
    feat=np.array([r(delta10) -r(U1)])
    feat=np.column_stack((feat,np.ones([feat.shape[0],1])))
    feat_help=np.array([r(delta10) -r(D1)])
    feat_help=np.column_stack((feat_help,2*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta10) -r(R1)])
    feat_help=np.column_stack((feat_help,3*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta10) -r(L1)])
    feat_help=np.column_stack((feat_help,4*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta20) -r(U2)])
    feat_help=np.column_stack((feat_help,5*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta20) -r(D2)])
    feat_help=np.column_stack((feat_help,6*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta20) -r(R2)])
    feat_help=np.column_stack((feat_help,7*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta20) -r(L2)])
    feat_help=np.column_stack((feat_help,8*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta30) -r(U3)])
    feat_help=np.column_stack((feat_help,9*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta30) -r(D3)])
    feat_help=np.column_stack((feat_help,10*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta30) -r(R3)])
    feat_help=np.column_stack((feat_help,11*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta30) -r(L3)])
    feat_help=np.column_stack((feat_help,12*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    #-----------
    feat_help=np.array([r(delta21) -r(U2)])
    feat_help=np.column_stack((feat_help,13*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta21) -r(D2)])
    feat_help=np.column_stack((feat_help,14*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta21) -r(R2)])
    feat_help=np.column_stack((feat_help,15*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta21) -r(L2)])
    feat_help=np.column_stack((feat_help,16*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=np.array([r(delta31) -r(U3)])
    feat_help=np.column_stack((feat_help,17*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta31) -r(D3)])
    feat_help=np.column_stack((feat_help,18*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta31) -r(R3)])
    feat_help=np.column_stack((feat_help,19*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta31) -r(L3)])
    feat_help=np.column_stack((feat_help,20*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=np.array([r(delta32) -r(U3)])
    feat_help=np.column_stack((feat_help,21*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta32) -r(D3)])
    feat_help=np.column_stack((feat_help,22*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta32) -r(R3)])
    feat_help=np.column_stack((feat_help,23*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(delta32) -r(L3)])
    feat_help=np.column_stack((feat_help,24*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=phi(dU10)
    feat_help=np.column_stack((feat_help,25*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dU20)
    feat_help=np.column_stack((feat_help,26*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dU30)
    feat_help=np.column_stack((feat_help,27*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

#---------
    feat_help=phi(dU21)
    feat_help=np.column_stack((feat_help,28*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dU31)
    feat_help=np.column_stack((feat_help,29*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=phi(dU32)
    feat_help=np.column_stack((feat_help,30*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))


    feat_help=phi(dD10)
    feat_help=np.column_stack((feat_help,31*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dD20)
    feat_help=np.column_stack((feat_help,32*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dD30)
    feat_help=np.column_stack((feat_help,33*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))


#------------
    feat_help=phi(dD21)
    feat_help=np.column_stack((feat_help,34*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dD31)
    feat_help=np.column_stack((feat_help,35*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dD32)
    feat_help=np.column_stack((feat_help,36*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))


    feat_help=phi(dR10)
    feat_help=np.column_stack((feat_help,37*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dR20)
    feat_help=np.column_stack((feat_help,38*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dR30)
    feat_help=np.column_stack((feat_help,39*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
#-------------
    feat_help=phi(dR21)
    feat_help=np.column_stack((feat_help,40*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dR31)
    feat_help=np.column_stack((feat_help,41*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dR32)
    feat_help=np.column_stack((feat_help,42*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=phi(dL10)
    feat_help=np.column_stack((feat_help,43*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dL20)
    feat_help=np.column_stack((feat_help,44*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dL30)
    feat_help=np.column_stack((feat_help,45*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
#--------------
    feat_help=phi(dL21)
    feat_help=np.column_stack((feat_help,46*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dL31)
    feat_help=np.column_stack((feat_help,47*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(dL32)
    feat_help=np.column_stack((feat_help,48*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=np.array([r(dU10)])
    feat_help=np.column_stack((feat_help,49*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dU20)])
    feat_help=np.column_stack((feat_help,50*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dU30)])
    feat_help=np.column_stack((feat_help,51*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

#-------
    feat_help=np.array([r(dU21)])
    feat_help=np.column_stack((feat_help,52*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dU31)])
    feat_help=np.column_stack((feat_help,53*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dU32)])
    feat_help=np.column_stack((feat_help,54*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    


    feat_help=np.array([r(dD10)])
    feat_help=np.column_stack((feat_help,55*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dD20)])
    feat_help=np.column_stack((feat_help,56*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dD30)])
    feat_help=np.column_stack((feat_help,57*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
#-------------
    feat_help=np.array([r(dD21)])
    feat_help=np.column_stack((feat_help,58*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dD31)])
    feat_help=np.column_stack((feat_help,59*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dD32)])
    feat_help=np.column_stack((feat_help,60*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=np.array([r(dR10)])
    feat_help=np.column_stack((feat_help,61*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dR20)])
    feat_help=np.column_stack((feat_help,62*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dR30)])
    feat_help=np.column_stack((feat_help,63*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
#------------------
    feat_help=np.array([r(dR21)])
    feat_help=np.column_stack((feat_help,64*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dR31)])
    feat_help=np.column_stack((feat_help,65*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dR32)])
    feat_help=np.column_stack((feat_help,66*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))

    feat_help=np.array([r(dL10)])
    feat_help=np.column_stack((feat_help,67*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dL20)])
    feat_help=np.column_stack((feat_help,68*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dL30)])
    feat_help=np.column_stack((feat_help,69*np.ones([feat_help.shape[0],1])))
#---------------------
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dL21)])
    feat_help=np.column_stack((feat_help,70*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dL31)])
    feat_help=np.column_stack((feat_help,71*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=np.array([r(dL32)])
    feat_help=np.column_stack((feat_help,72*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))


    feat_help=phi(a)
    feat_help=np.column_stack((feat_help,73*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(b)
    feat_help=np.column_stack((feat_help,74*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(c)
    feat_help=np.column_stack((feat_help,75*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    feat_help=phi(d)
    feat_help=np.column_stack((feat_help,76*np.ones([feat_help.shape[0],1])))
    feat=np.row_stack((feat,feat_help))
    return feat