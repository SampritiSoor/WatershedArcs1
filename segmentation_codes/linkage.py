import scipy.cluster.hierarchy as hac
import numpy as np

def getLinkageSeg(img,Method,nClusters=2,positionAsFeature=False):
    def myResize(image):
        if image.shape[0]>250:
            image=image[::2,:]
        if image.shape[1]>250:
            image=image[:,::2]
        return image

    def myInvResize(image,h,w):
        if image.shape[0]<h:
            image=np.repeat(image, 2, axis=0)
        if image.shape[0]>h:
            image=image[:-1,:]

        if image.shape[1]<w:
            image=np.repeat(image, 2, axis=1)
        if image.shape[1]>w:
            image=image[:,:-1]

        return image

    def indices_array_generic(m,n):
        r0 = np.arange(m) # Or r0,r1 = np.ogrid[:m,:n], out[:,:,0] = r0
        r1 = np.arange(n)
        out = np.empty((m,n,2),dtype=int)
        out[:,:,0] = r0[:,None]
        out[:,:,1] = r1
        return out

    def getZ(image,Method,positionAsFeature=False):
        h,w=image.shape[0],image.shape[1]
        if positionAsFeature:
            ind=indices_array_generic(h,w)
            img = np.expand_dims(image, axis=2)
            arr=np.concatenate((img,ind),axis=2)
            arr_rshp=arr.reshape(h*w,3)
        else:
            arr_rshp=image.reshape(h*w,1)
        return hac.linkage(arr_rshp,method=Method)

    def get_hierSeg_Linkage(z,nClusters=2):
        label=hac.fcluster(z,t=nClusters,criterion='maxclust')
        return label
    
    h_0,w_0=img.shape[0],img.shape[1]
    
    img=myResize(img)
    h_1,w_1=img.shape[0],img.shape[1]
    
    z=getZ(img,Method,positionAsFeature=False)

    segments=get_hierSeg_Linkage(z,nClusters=2)
    segments=myInvResize(segments.reshape(h_1,w_1),h_0,w_0)
    return segments