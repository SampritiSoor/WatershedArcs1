import numpy as np
import itertools
from skimage.morphology import reconstruction

def VS_watershed(image):
    def get8ConnList(shape):
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
            N[i*ww+j]=[int(nbrs[k][i][j]/shape[1])*ww+nbrs[k][i][j]%shape[1] for k in range(8) if nbrs[k][i][j]>=0]
        return N

    ht,wd=image.shape
    N8=get8ConnList((ht,wd))
    hDict={}
    for h in range(ht):
        for w in range(wd):
            if hDict.get(image[h,w]) is None:
                hDict[image[h,w]]=[h*wd+w]
            else:
                hDict[image[h,w]].append(h*wd+w)

    hList=list(hDict.keys())
    hList.sort()

    label=[-1]*(ht*wd) # -1: init ; -2: mask ; 0: wshed
    dist=[0]*(ht*wd)
    queue=[]
    curlab=0

    for h in hList:
        for p in hDict[h]:
            label[p]=-2
    #         print(p,len([1 for n in N8[p] if label[n]>=0]))
            if len([1 for n in N8[p] if label[n]>=0])>0:
                dist[p]=1
                queue.append(p)

    #     print("aa",label)
    #     print(queue)
        currdist=1
        queue.append(-1) # -1 : fictitous
        while True:
            p=queue.pop(0)
            if p==-1:
                if len(queue)==0:
                    break
                else:
                    queue.append(-1)
                    currdist+=1
                    p=queue.pop(0)

    #         print(p)
            for q in N8[p]:
                if dist[q]<currdist and label[q]>=0:
                    if label[q]>0:
                        if label[p]==-2 or label[p]==0:
                            label[p]=label[q]
                        elif label[p]!=label[q]:
                            label[p]=0
                    elif label[p]==-2:
                        label[p]=0
                elif label[q]==-2 and dist[q]==0:
                    dist[q]=currdist+1
                    queue.append(q)

        for p in hDict[h]:
            dist[p]=0
            if label[p]==-2:
                curlab+=1
                queue.append(p)
                label[p]=curlab
                while len(queue)>0:
                    q=queue.pop(0)
                    for r in N8[q]:
                        if label[r]==-2:
                            queue.append(r)
                            label[r]=curlab
    return label

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

def get_hierSeg_Waterfall(image):
    hier_seg,hier_boun=[],[]
    if image.ndim!=2:
        print("Waterfall Algorithm works for 2-D images")
        return hier_seg,hier_boun
    else:
        label=VS_watershed(image)
        vs_label=(np.array(label)).reshape(image.shape)
        hier_seg.append(vs_label)
        vs_ws=np.where(vs_label==0,1,0)
        while np.sum(vs_ws)>0:
            y_seed=np.where(vs_label==0,image,255)
            y_mask = image
            image = reconstruction(y_seed,y_mask,method='erosion').astype('int')
            label=VS_watershed(image)
            vs_label=(np.array(label)).reshape(image.shape)
            hier_seg.append(getWSseg(vs_label))
            vs_ws=np.where(vs_label==0,1,0)
            hier_boun.append(vs_ws)
        return hier_seg,hier_boun