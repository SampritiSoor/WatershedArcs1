import itertools
import numpy as np
import random
from queue import PriorityQueue


def getIWseg(img,nClusters=3,maxIter=5,initCenters=None,distType='direct',doPrint=False):    
    def euclDist(v1,v2):
        if imgChannels==1:
            return np.abs(np.int(v1)-np.int(v2))
        else:
            return np.linalg.norm(np.array(v1) - np.array(v2)) 
    def getConnList(shape,conn=8,frame=None):
        hh,ww=shape[0],shape[1]
        basis=np.arange(hh*ww).reshape(hh,ww)
        nbrs=[]
        nbrs.append(np.append(np.full((1,ww),-1),basis[:-1],axis=0)) #top
        nbrs.append(np.append(np.full((1,hh),-1).T,basis[:,:-1],axis=1)) #left
        nbrs.append(np.append(basis[:,1:],np.full((1,hh),-1).T,axis=1)) #right
        nbrs.append(np.append(basis[1:],np.full((1,ww),-1),axis=0)) #bottom

        nbrs.append(np.append(np.full((1,ww),-1),(np.append(np.full((1,hh-1),-1).T,basis[:-1,:-1],axis=1)),axis=0)) #top-left
        nbrs.append(np.append((np.append(basis[1:,1:],np.full((1,hh-1),-1).T,axis=1)),np.full((1,ww),-1),axis=0)) #bottom-right
        nbrs.append(np.append(np.full((1,ww),-1),(np.append(basis[:-1,1:],np.full((1,hh-1),-1).T,axis=1)),axis=0)) #top-right
        nbrs.append(np.append((np.append(np.full((1,hh-1),-1).T,basis[1:,:-1],axis=1)),np.full((1,ww),-1),axis=0)) #bottom-left

        N={}
        for i, j in itertools.product(range(hh), range(ww)):
            if frame is None:
                N[(i,j)]=[(int(nbrs[k][i][j]/shape[1]),nbrs[k][i][j]%shape[1]) for k in range(conn) if nbrs[k][i][j]>=0]
            else:
                N[(i,j)]=[(int(nbrs[k][i][j]/shape[1]),nbrs[k][i][j]%shape[1]) for k in range(conn) if nbrs[k][i][j]>=0 and frame[(int(nbrs[k][i][j]/shape[1]),nbrs[k][i][j]%shape[1])]==0]
        return N
    N=getConnList(img.shape,conn=4) 
    imgChannels=1 if img.ndim<=2 else img.shape[-1]
    
    def getCluster(centers,doPrint=False):
        changed={}
        status={(i,j):'undiscovered' for i in range(img.shape[0]) for j in range(img.shape[1])}
        currentDist={(i,j):np.Inf for i in range(img.shape[0]) for j in range(img.shape[1])}
        currentLabel={(i,j):np.Inf for i in range(img.shape[0]) for j in range(img.shape[1])}
        Q=PriorityQueue()
        for c in range(len(centers)):
            Q.put((0,centers[c],c))
            status[centers[c]]='inQ'
            currentDist[centers[c]]=0
            currentLabel[centers[c]]=c
    #     if doPrint: print("Q",Q.queue)

        while not Q.empty():
            dist_p,p,label_p=Q.get()
    #         if doPrint: 
    #             print("dist_p,p,label_p",dist_p,p,label_p)
    #             print("Q",Q.queue)
            if changed.get((dist_p,p)) is not None:
    #             if doPrint: print('continue')
                continue
            status[p]='popped'
    #         if doPrint: print('status',status)
            for n in N[p]:
                if doPrint: print('   n',n)
                insertInQ=False
                if status[n]!='popped':
                    if distType=='max':
                        d=max(currentDist[p],euclDist(img[p],img[n]))
                    elif distType=='direct':
                        d=euclDist(img[centers[currentLabel[p]]],img[n])
    #                 if doPrint: print('   d',d)
                    if status[n]=='undiscovered':
    #                     if doPrint: print('   undiscovered')
                        status[n]='inQ'
                        insertInQ=True
                    elif status[n]=='inQ':
                        if d<currentDist[n]:
                            if doPrint: print('   inQ changed')
                            changed[(currentDist[n],n)]='dummy'
                            insertInQ=True
    #                     else: 
    #                         if doPrint: print('   inQ')
                if insertInQ:
                    Q.put((d,n,label_p))
                    currentDist[n]=d
                    currentLabel[n]=label_p
    #         if doPrint: print("Q",Q.queue)
        return currentLabel

    if initCenters is None:
        data=img.flatten()
        argSrt=np.argsort(data)
        minArg=np.argsort(data)[0]
        maxArg=np.argsort(data)[-1]
        medianArg=np.argsort(data)[len(data)//2]
        minPos=(int(minArg/img.shape[1]),minArg%img.shape[1])
        maxPos=(int(maxArg/img.shape[1]),maxArg%img.shape[1])
        medianPos=(int(medianArg/img.shape[1]),medianArg%img.shape[1])
        if nClusters==2: 
            centers=[minPos,maxPos] 
        if nClusters==3: 
            centers=[minPos,maxPos,medianPos]
        else:
            count=0
            centers=[]
            while count<nClusters:
                p=(random.randrange(img.shape[0]),random.randrange(img.shape[1]))
                centers.append(p)
                count+=1
    else:
        centers=initCenters
        
#     if doPrint: 
#         print("init centers",centers) 
#         print("init values",[img[r] for r in centers])      

    itr=0
    prevCenters=[centers]
    while True:
        if doPrint: print("itr",itr)
        label=getCluster(centers,doPrint=False)
#         if doPrint:
#             segImage=np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
#             for i in range(img.shape[0]):
#                 for j in range(img.shape[1]):
#                     segImage[i,j]=label[(i,j)]
#             print(segImage)

        itr+=1
        if itr==maxIter:
            break

        clustersize=np.zeros(nClusters)
        cluster_sumFeatures=np.zeros((nClusters,img.shape[-1])) if img.ndim==3 else np.zeros(nClusters)
        clusters=[[] for i in range(nClusters)]
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                cluster_sumFeatures[label[(i,j)]]+=img[i,j]
                clustersize[label[(i,j)]]+=1
                clusters[label[(i,j)]].append((i,j))   
        cluster_meanFeatures=np.divide(cluster_sumFeatures,np.expand_dims(clustersize,axis=1)) if img.ndim==3 else cluster_sumFeatures/clustersize
#         if doPrint: 
#             print("cluster_sumFeatures",cluster_sumFeatures)    
#             print("cluster_meanFeatures",cluster_meanFeatures)             

        centers=[]
        for i in range(nClusters):
            dists=[]
            for c in clusters[i]:
                dists.append(euclDist(cluster_meanFeatures[i],img[c]))
            k=np.argmin(dists)
            centers.append( clusters[i][k] ) 
            
        if set(centers)==set(prevCenters[-1]):
            break
        else:
            if len(prevCenters)>1 and set(centers)==set(prevCenters[-2]):
                break
        prevCenters.append(centers)

#         if doPrint:
#             print("next centers",centers) 
#             print("init values",[img[r] for r in centers])   
#             print("---------------------------------------------------------")

    segImage=np.zeros((img.shape[0],img.shape[1]),dtype=np.int32)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            segImage[i,j]=label[(i,j)]+1
    return segImage