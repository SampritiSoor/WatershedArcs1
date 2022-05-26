import numpy as np
import cv2
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import adjusted_mutual_info_score

def getFB(methodBoundary,gtBoundary):
    def getfob2(I1,I2):
        uniImage=np.where(I1+I2>0, True, False)
        I1_flat,I2_flat=[],[]
        for i in range(uniImage.shape[0]):
            for j in range(uniImage.shape[1]):
                if uniImage[i,j]:
                    I1_flat.append(I1[i,j])
                    I2_flat.append(I2[i,j])
        return f1_score(np.array(I1_flat),np.array(I2_flat))
    resultBoundaryH=[]
    kernels=[]
    kernels.append(np.array([[0,0,0],[1,1,0],[1,1,0]], np.uint8))
    kernels.append(np.array([[0,1,1],[0,1,1],[0,0,0]], np.uint8))
    kernels.append(np.array([[1,1,0],[1,1,0],[0,0,0]], np.uint8))
    kernels.append(np.array([[0,0,0],[0,1,1],[0,1,1]], np.uint8))
    for kernel in kernels:
        resultBoundaryH.append(cv2.morphologyEx(methodBoundary.astype(np.uint8), cv2.MORPH_DILATE, kernel))
    maxQualMeasure=-np.Inf
    for B in resultBoundaryH:
        thisQualMeasure=getfob2(B,gtBoundary)
    if maxQualMeasure<thisQualMeasure:
        maxQualMeasure=thisQualMeasure
    return maxQualMeasure

def getSliceLabel(image):
    def getNeighbours8(p,shp):
        N=[]
        for i in range(p[0]-1,p[0]+2):
            if i<0: continue
            if i>=shp[0]: continue
            for j in range(p[1]-1,p[1]+2):
                if j<0: continue
                if j>=shp[1]: continue
                if i==p[0] and j==p[1]: continue
                N.append((i,j))
        return N
    shp=image.shape
    bSegImage=np.zeros(shp,dtype=np.int32)
    toSegMask=np.full(shp,True)
    segNo=0
    for i in range(shp[0]):
        for j in range(shp[1]):
            if not toSegMask[i,j]: continue
            segNo+=1
            thisLabel=image[i,j]
            Q=[(i,j)]
            toSegMask[i,j]=False
            bSegImage[i,j]=segNo
            while len(Q)>0:
                q=Q.pop(0)
                for n in getNeighbours8(q,shp):
                    if toSegMask[n] and image[n]==thisLabel:
                        Q.append(n)
                        toSegMask[n]=False
                        bSegImage[n]=segNo
    return bSegImage

from skimage.morphology import diamond,square
def getFOBP(res,gt,doPrint=False,doWrite=False,gamma_o=.7,gamma_p=.1,beta=.5):
    if (np.max(res)-np.min(res))==1:
        if np.max(res)>1: res=res-(np.max(res)-1)
        res=res.astype(np.uint8)
        res= cv2.morphologyEx(res, cv2.MORPH_OPEN, diamond(2))
        res= cv2.morphologyEx(res, cv2.MORPH_CLOSE, diamond(2))
    res=getSliceLabel(res)
    gt=getSliceLabel(gt)
    S=[np.where(res==v,True,False) for v in range(np.min(res),np.max(res)+1)]
    G=[np.where(gt==v,True,False) for v in range(np.min(gt),np.max(gt)+1)]
    if doPrint: print(len(S),len(G))
    if doWrite:
        for s in range(len(S)):
            cv2.imwrite('S_'+str(s)+'.png', (S[s]).astype(np.uint8)*255)
        for g in range(len(G)):
            cv2.imwrite('G_'+str(g)+'.png', (G[g]).astype(np.uint8)*255)
    Oij_S,Oij_G={},{}
    oc,oc_,pc,pc_,fc,fc_,nc,nc_=0,0,0,0,0,0,0,0
    markS,markG={},{}
    for i in range(len(S)):
        for j in range(len(G)):
            Oij_S[(i,j)]= np.sum(S[i]&G[j])/np.sum(S[i])
            Oij_G[(i,j)]= np.sum(S[i]&G[j])/np.sum(G[j])
#             print("Oij_S[(i,j)],Oij_G[(i,j)]",Oij_S[(i,j)],Oij_G[(i,j)])
            
            if Oij_S[(i,j)]>gamma_o and Oij_G[(i,j)]>gamma_o:
                oc+=1
                oc_+=1
                markS[(i,j)],markG[(i,j)]='o','o'
            elif Oij_S[(i,j)]>gamma_p and Oij_G[(i,j)]>gamma_o:
                fc+=1
                pc_+=1
                markS[(i,j)],markG[(i,j)]='f','p'
            elif Oij_S[(i,j)]>gamma_o and Oij_G[(i,j)]>gamma_p:
                pc+=1
                fc_+=1
                markS[(i,j)],markG[(i,j)]='p','f'
            else:
                nc+=1
                nc_+=1
                markS[(i,j)],markG[(i,j)]='n','n'
#             break
#         break
            if doPrint:
                print("Si,Gj",i,j)
                print("   oc,oc_",oc,oc_,)
                print("   pc,pc_",pc,pc_)
                print("   fc,fc_",fc,fc_)
                print("   nc,nc_",nc,nc_)
    if doPrint:        
        print("Oij_S",Oij_S)
        print("Oij_G",Oij_G)
        print("markS",markS)
        print("markG",markG)
    
    fr=np.sum([np.sum([ Oij_G[(i,j)] for j in range(len(G)) if  Oij_S[(i,j)]>gamma_o and markG[(i,j)]=='f']) for i in range(len(S))])
    fr_=np.sum([np.sum([ Oij_S[(i,j)] for j in range(len(G)) if  Oij_G[(i,j)]>gamma_o and markS[(i,j)]=='f']) for i in range(len(S))])
#     fr,fr_=0,0.92
    if doPrint:
        print("oc/S",oc/len(S),"pc/S",pc/len(S),"oc_/G",oc_/len(G),"pc_/G",oc_/len(G))
        print("fr,fr_",fr,fr_)
    P_op=(oc+fr+beta*pc)/len(S)
    R_op=(oc_+fr_+beta*pc_)/len(G)
    
    if doPrint:
        print("P_op",P_op)
        print("R_op",R_op)
        
    if P_op+R_op==0:
        return 0
    else:
        return(2*P_op*R_op)/(P_op+R_op)

def nC2(n):
    return (n*(n-1))/2
def getARI(L1,L2):
    L1=L1.flatten()
    L2=L2.flatten()
    Labels_L1=list(set(L1))
    posL1={x:i for i,x in enumerate(Labels_L1)}
    Labels_L2=list(set(L2))
    posL2={x:i for i,x in enumerate(Labels_L2)}
    mat=np.zeros((len(Labels_L1),len(Labels_L2)))
    for l in range(len(L1)):
        mat[posL1[L1[l]],posL2[L2[l]]]+=1
    FT=sum([nC2(mat[i,j]) for i in range(mat.shape[0]) for j in range(mat.shape[1])])
    ST=sum([nC2(sum(mat[i,:]))*nC2(sum(mat[:,j]) ) for i in range(mat.shape[0]) for j in range(mat.shape[1])])/nC2(len(L1))
    TT=(sum([nC2(sum(mat[i,:])) for i in range(mat.shape[0])])+sum([nC2(sum(mat[:,j])) for j in range(mat.shape[1])]))/2
    return (FT-ST)/(TT-ST)

def getSeg(bounImage):
    def getWSseg(image):
        shp=image.shape
        wsSegImage=np.copy(image)
        toSegMask=np.where(image==0,True,False)
        for i in range(shp[0]):
            for j in range(shp[1]):
                if toSegMask[i,j]:
                    if j+1<shp[1] and not toSegMask[i,j+1]:
                        wsSegImage[i,j]=image[i,j+1]
                    elif i+1<shp[0] and not toSegMask[i+1,j]:
                        wsSegImage[i,j]=image[i+1,j]
                    elif j-1>=0 and not toSegMask[i,j-1]:
                        wsSegImage[i,j]=image[i,j-1]
                    elif i-1>=0 and not toSegMask[i-1,j]:
                        wsSegImage[i,j]=image[i-1,j]
        return wsSegImage
    def getBasinSeg(image,conn=4):
        if conn!=8:
            if conn!=4:
                conn=8
        shp=image.shape
        bSegImage=np.zeros(shp,dtype=np.int32)
        toSegMask=np.where(image==1,False,True)
        segNo=0
        for i in range(shp[0]):
            for j in range(shp[1]):
                if not toSegMask[i,j]: continue
                segNo+=1
                Q=[(i,j)]
                toSegMask[i,j]=False
                bSegImage[i,j]=segNo
                while len(Q)>0:
                    q=Q.pop(0)
                    for n in getNeighbours(q,shp,conn):
                        if toSegMask[n]:
                            Q.append(n)
                            toSegMask[n]=False
                            bSegImage[n]=segNo
        return bSegImage
    def getNeighbours(p,shp,conn):
        if conn==8:
            N=[]
            for i in range(p[0]-1,p[0]+2):
                if i<0: continue
                if i>=shp[0]: continue
                for j in range(p[1]-1,p[1]+2):
                    if j<0: continue
                    if j>=shp[1]: continue
                    if i==p[0] and j==p[1]: continue
                    N.append((i,j))
            return N
        if conn==4:
            N=[]
            for i in range(p[0]-1,p[0]+2):
                if i<0: continue
                if i>=shp[0]: continue
                for j in range(p[1]-1,p[1]+2):
                    if j<0: continue
                    if j>=shp[1]: continue
                    if i==p[0] and j==p[1]: continue
                    if i==p[0] or j==p[1]: N.append((i,j))
            return N

    return getWSseg(getBasinSeg(bounImage))

def getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='AMI'):
    maxQualMeasure=-np.Inf
    bestMatchedGtIndex=np.Inf
    for g in range(len(gtObj)):
        if evalPriority=='AMI':
            thisQualMeasure=adjusted_mutual_info_score(method_seg.flatten(),gtObj[g].flatten())
        elif evalPriority=='ARI':
            thisQualMeasure=getARI(method_seg,gtObj[g])
        elif evalPriority=='FOP':
            thisQualMeasure=getFOBP(method_seg,gtObj[g])
        else:
            thisQualMeasure=getFB(method_boun,gtBoun[g])
            
        if maxQualMeasure<thisQualMeasure:
            maxQualMeasure=thisQualMeasure
            bestMatchedGtIndex=g
            
    fb=getFB(method_boun,gtBoun[bestMatchedGtIndex])
    fop=getFOBP(method_seg,gtObj[bestMatchedGtIndex])
    ami=adjusted_mutual_info_score(method_seg.flatten(),gtObj[bestMatchedGtIndex].flatten())
    ari=getARI(method_seg,gtObj[bestMatchedGtIndex])
    
    return bestMatchedGtIndex, fb, fop, ami, ari