import numpy as np
import cv2
from sklearn.metrics.cluster import adjusted_mutual_info_score

def getFB(res,gt,FNweight=.5):
    res=res.astype(int)
    gt=gt.astype(int)
    if np.sum(gt)==0:
        return 0
    mapped=np.zeros(gt.shape)
    TP,TN,FP,FN=0,0,0,0
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            if res[i,j]==1:
                if i-1>=0 and gt[i-1,j]==1 and mapped[i-1,j]==0:
                    TP+=1
                    mapped[i-1,j]=1
                elif j-1>=0 and gt[i,j-1]==1 and mapped[i,j-1]==0:
                    TP+=1
                    mapped[i,j-1]=1
                elif gt[i,j]==1 and mapped[i,j]==0:
                    TP+=1
                    mapped[i,j]=1
                elif j+1<gt.shape[1] and gt[i,j+1]==1 and mapped[i,j+1]==0:
                    TP+=1
                    mapped[i,j+1]=1
                elif i+1<gt.shape[0] and gt[i+1,j]==1 and mapped[i+1,j]==0:
                    TP+=1
                    mapped[i+1,j]=1
                else:
                    FP+=1
    FN=max(np.sum(gt)*FNweight-TP,0)
    return TP/(TP+.5*(FP+FN))

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
def getFOP(res,gt,doPrint=False,doWrite=False,gamma_o=.7,gamma_p=.1,beta=.5,binaryLabel=False):
    if binaryLabel:
        gt=np.where(gt==1,1,0)

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
    R_op=fr/(fc_*len(G)) if (oc_+fr_+beta*pc_)==0 else (oc_+fr_+beta*pc_)/len(G)
    
    if doPrint:
        print("P_op",P_op)
        print("R_op",R_op)
        
    if P_op+R_op==0:
        return 0
    else:
        return(2*P_op*R_op)/(P_op+R_op)

def nC2(n):
    return (n*(n-1))/2

def ariCalc(res,gt):
    res=res.flatten()
    gt=gt.flatten()
    Labels_res=list(set(res))
    posres={x:i for i,x in enumerate(Labels_res)}
    Labels_gt=list(set(gt))
    posgt={x:i for i,x in enumerate(Labels_gt)}
    mat=np.zeros((len(Labels_res),len(Labels_gt)))
    for l in range(len(res)):
        mat[posres[res[l]],posgt[gt[l]]]+=1
    FT=sum([nC2(mat[i,j]) for i in range(mat.shape[0]) for j in range(mat.shape[1])])
    ST=sum([nC2(sum(mat[i,:]))*nC2(sum(mat[:,j]) ) for i in range(mat.shape[0]) for j in range(mat.shape[1])])/nC2(len(res))
    TT=(sum([nC2(sum(mat[i,:])) for i in range(mat.shape[0])])+sum([nC2(sum(mat[:,j])) for j in range(mat.shape[1])]))/2
    return (FT-ST)/(TT-ST)
    
def getARI(res,gt,binaryLabel=False):
    if binaryLabel:
        gt=np.where(gt==1,1,0)
        return ariCalc(res,gt)
    else:
        ari1=ariCalc(res,gt)
        gt_sl=getSliceLabel(gt)
        if np.max(gt_sl)-np.min(gt_sl)==np.max(gt)-np.min(gt): return ari1
        else: return max(ari1,ariCalc(res,gt_sl))




def getSeg(bounImage,frame=None,segBoundary=False):
    if frame is not None: frame=cv2.morphologyEx(frame.astype(np.uint8), cv2.MORPH_ERODE, np.ones((3,3))).astype(np.bool)
    def getWSseg(image):
        shp=image.shape
        wsSegImage=np.copy(image)
        toSegMask=np.where(image==0,True,False)
        if frame is not None: toSegMask=toSegMask&frame
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
    def getBasinSeg(image):
        shp=image.shape
        bSegImage=np.zeros(shp,dtype=np.int32)
        toSegMask=np.where(image==1,False,True)
        if frame is not None: toSegMask=toSegMask&frame
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
                    for n in get4Neighbours(q,shp):
                        if toSegMask[n]:
                            Q.append(n)
                            toSegMask[n]=False
                            bSegImage[n]=segNo
        return bSegImage
    def get4Neighbours(p,shp):
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
    
    if segBoundary:
        return getBasinSeg(bounImage)
    else:
        return getWSseg(getBasinSeg(bounImage))


def getAMI(res,gt,binaryLabel=False):
    if binaryLabel:
        gt=np.where(gt==1,1,0)   
        return adjusted_mutual_info_score(res.flatten(),gt.flatten())
    else:
        ami1=adjusted_mutual_info_score(res.flatten(),gt.flatten())
        gt_sl=getSliceLabel(gt)
        if np.max(gt_sl)-np.min(gt_sl)==np.max(gt)-np.min(gt): return ami1
        else: return max(ami1,adjusted_mutual_info_score(res.flatten(),gt_sl.flatten()))

def getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='AMI',gamma_o=.75,gamma_p=.1,beta=.5,binaryLabel=False,FNweight=.5,doPrint=False,doWrite=False):
    maxQualMeasure=-np.Inf
    bestMatchedGtIndex=np.Inf
    for g in range(len(gtObj)):
        if evalPriority=='AMI':
            thisQualMeasure=getAMI(method_seg,gtObj[g],binaryLabel=binaryLabel)
        elif evalPriority=='ARI':
            thisQualMeasure=getARI(method_seg,gtObj[g],binaryLabel=binaryLabel)
        elif evalPriority=='FOP':
            thisQualMeasure=getFOP(method_seg,gtObj[g],binaryLabel=binaryLabel,gamma_o=gamma_o,gamma_p=gamma_p,beta=beta,doPrint=doPrint,doWrite=doWrite)
        else:
            thisQualMeasure=getFB(method_boun,gtBoun[g],FNweight=FNweight)
            
        if maxQualMeasure<thisQualMeasure:
            maxQualMeasure=thisQualMeasure
            bestMatchedGtIndex=g
            
    fb=getFB(method_boun,gtBoun[bestMatchedGtIndex],FNweight=FNweight)
    fop=getFOP(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel,gamma_o=gamma_o,gamma_p=gamma_p,beta=beta,doPrint=doPrint,doWrite=doWrite)
    ami=getAMI(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel)
    ari=getARI(method_seg,gtObj[bestMatchedGtIndex],binaryLabel=binaryLabel)
    
    return bestMatchedGtIndex, fb, fop, ami, ari

def get_argmaxBestPartFrac(resFiles,gt):
    G=[np.where(gt==i,True,False) for i in range(np.min(gt),np.max(gt)+1)]

    hLevels,partFracs=[],[]
    for r in range(len(resFiles)):
        res=resFiles[r]
        S=[np.where(res==i,True,False) for i in range(np.min(res),np.max(res)+1)]
        hLevels.append(len(S))
        
        if hLevels[-1]>=len(G):
            partIntersections=[]
            for g in G:
                partIntersections.append([(np.round(np.sum(S[s]&g)/np.sum(g),4),s) for s in range(len(S))]) # if ((S[s]&g)==S[s]).all()

            for PI in partIntersections:
                PI.sort(reverse=True)
            gtFixed={i:False for i in range(len(G))}
            gtFixedCount=0
            segFixed={i:False for i in range(hLevels[-1])}
            partFrac=0
            for i in range(len(segFixed)):
                thisGtParts=[PI[i] for PI in partIntersections]
                segNos=np.arange(len(G))
                thisGtParts, segNos = zip(*sorted(zip(thisGtParts, segNos),reverse=True))
#                 print(thisGtParts,segNos)
                
                for t in range(len(G)):
                    if not gtFixed[segNos[t]]:
                        if not segFixed[thisGtParts[t][1]]:
                            partFrac+=thisGtParts[t][0]
                            segFixed[thisGtParts[t][1]]=True
                            gtFixed[segNos[t]]=True
                            gtFixedCount+=1
                            break
                if len(G)==gtFixedCount: break
            if r>5 and partFrac<partFracs[-1]:
                return r-1
            partFracs.append(partFrac)
    return np.argmax(partFracs)