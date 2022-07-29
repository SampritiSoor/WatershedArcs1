import cv2
import os
import numpy as np
from skimage.segmentation import find_boundaries
import time

from skimage.segmentation import morphological_chan_vese
from sklearn.metrics.cluster import adjusted_mutual_info_score

from segmentation_codes.Warcs_removal import get_hierSeg_WSarcs_arcRemove
from segmentation_codes.waterfall import get_hierSeg_Waterfall
from segmentation_codes.iteratedWatershed import getIWseg
from segmentation_codes.linkage import getLinkageSeg
from segmentation_codes.evaluationMetrics import getSeg,getEvalScores

import matplotlib.pyplot as plt

from skimage.morphology import diamond,square
def morphoFilter(image,oper): 
    if oper[2]=='s':
        kernel=square(int(oper[1]))
    elif oper[2]=='d':
        kernel=diamond(int(oper[1]))
    
    if oper[0]=='E':
        return cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
    elif oper[0]=='D':
        return cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
    elif oper[0]=='O':
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    elif oper[0]=='C':
        return cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    elif oper[0]=='G':
        return cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
    elif oper[0]=='S':
        return cv2.filter2D(image, -1, sharpen_filter)

def generateEvaluationPlot(imageName,noOfObj,dpi):
    img=cv2.imread(os.path.join('sample_Wiezmann_data',str(noOfObj)+'obj',imageName,'src_bw',imageName+'.png'),0)

    dispSize = 1.35*img.shape[1] / float(dpi), 1.35*img.shape[0] / float(dpi)

    row,col=1,2
    fig, (ax1, ax2) = plt.subplots(row,col,figsize=(dispSize[0]*col,dispSize[1]*row))

    fig.suptitle(str(noOfObj)+'obj - '+imageName+'\n', fontsize=20)

    ax1.set_title('Original Image')
    ax1.imshow(img,cmap='gray')

    img_prep=img
    method='O3sC2dG5s'
    for m in range(0,len(method),3):
        img_prep=morphoFilter(img_prep,method[m:m+3])

    ax2.set_title('preprocess - '+method)
    ax2.imshow(img_prep,cmap='gray')

    plt.show()

    gtfileNames=os.listdir(os.path.join('sample_Wiezmann_data',str(noOfObj)+'obj',imageName,'human_seg'))
    row,col=2,len(gtfileNames)
    fig, ax = plt.subplots(row,col,figsize=(dispSize[0]*col,dispSize[1]*row))

    gtObj,gtBoun=[],[]
    for n in range(len(gtfileNames)):
        thisObjImg=cv2.imread(os.path.join('sample_Wiezmann_data',str(noOfObj)+'obj',imageName,'human_seg',gtfileNames[n]))
        ax[0][n].imshow(cv2.cvtColor(thisObjImg, cv2.COLOR_BGR2RGB))
        ax[0][n].set_title('Ground Truth - '+str(n+1)+' : '+gtfileNames[n].split('.')[0]+'\n\nobject')


        gtSeg=np.ones((thisObjImg.shape[0],thisObjImg.shape[1]),dtype=np.int)
        for i in range(thisObjImg.shape[0]):
            for j in range(thisObjImg.shape[1]):
                if thisObjImg[i,j,0]==thisObjImg[i,j,1]==thisObjImg[i,j,2]: continue
                else:gtSeg[i,j]=2
        gtObj.append(gtSeg)

        boun=find_boundaries(gtSeg, mode='thick')
        gtBoun.append(boun)
        ax[1][n].imshow(boun*255,cmap='gray')
        ax[1][n].set_title('boundary')

    # plt.title()
    plt.show()

    def showResultPlot(method_boun,method_seg,bestMatchedGtIndex, fb, fop, ami, ari, t, methodName=''):
        dispSize = 1.5*method_seg.shape[1] / float(dpi), 1.5*method_seg.shape[0] / float(dpi)
        row,col=1,3
        fig, (ax1, ax2, ax3) = plt.subplots(row,col,figsize=(dispSize[0]*col,dispSize[1]*row))
        ax1.imshow(method_boun,cmap='gray')
        ax1.set_title('boundary')
        ax2.imshow(method_seg,cmap='plasma')
        ax2.set_title('segments')
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.axis('off')
        ax3.text(0,.85,methodName,size=15, weight='bold')
        ax3.text(0,.7,'Run-time : '+str(np.round(t,2))+' sec',size=12)
        ax3.text(0,.6,'best matched : GT-'+str(bestMatchedGtIndex+1)+' '+gtfileNames[bestMatchedGtIndex].split('.')[0],size=12)
        ax3.text(0,.5,'FB score : '+str(np.round(fb,4)),size=12)
        ax3.text(0,.4,'FOP score : '+str(np.round(fop,4)),size=12)
        ax3.text(0,.3,'ARI score : '+str(np.round(ari,4)),size=12)
        ax3.text(0,.2,'AMI score : '+str(np.round(ami,4)),size=12)
        plt.show()

    t=time.time()
    Hierarchies_WAR=get_hierSeg_WSarcs_arcRemove(img_prep,nhood=8)
    t=time.time()-t
    segIndex=-1
    method_boun=(255-Hierarchies_WAR[segIndex])/255
    method_seg=getSeg(method_boun)
    bestMatchedGtIndex, fb, fop, ami, ari = getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='ARI')
    showResultPlot(method_boun,method_seg,bestMatchedGtIndex, fb, fop, ami, ari, t, methodName='Watershed Arcs Removal')

    t=time.time()
    segments_hier_waterfall, boundary_hier_waterfall = get_hierSeg_Waterfall(img_prep)
    t=time.time()-t
    method_boun=boundary_hier_waterfall[-2]
    method_seg=segments_hier_waterfall[-2]
    bestMatchedGtIndex, fb, fop, ami, ari = getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='ARI')
    showResultPlot(method_boun,method_seg,bestMatchedGtIndex, fb, fop, ami, ari, t, methodName='Waterfall')

    t=time.time()
    method_seg=getLinkageSeg(img,'ward')
    t=time.time()-t
    method_boun=find_boundaries(method_seg, mode='thick')

    bestMatchedGtIndex, fb, fop, ami, ari = getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='ARI')
    showResultPlot(method_boun,method_seg,bestMatchedGtIndex, fb, fop, ami, ari, t, methodName='Ward\'s linkage')

    t=time.time()
    method_seg=getLinkageSeg(img,'weighted')
    t=time.time()-t
    method_boun=find_boundaries(method_seg, mode='thick')
    bestMatchedGtIndex, fb, fop, ami, ari = getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='ARI')
    showResultPlot(method_boun,method_seg,bestMatchedGtIndex, fb, fop, ami, ari, t, methodName='Weighted linkage')

    t=time.time()
    method_seg = morphological_chan_vese(img,
                                       num_iter=250,
                                       init_level_set='checkerboard',
                                       smoothing=1, lambda1=1, lambda2=1)
    t=time.time()-t
    method_boun=find_boundaries(method_seg, mode='thick')
    bestMatchedGtIndex, fb, fop, ami, ari = getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='ARI')
    showResultPlot(method_boun,method_seg,bestMatchedGtIndex, fb, fop, ami, ari, t, methodName='morphACWE')

    t=time.time()
    n=2 if noOfObj==1 else 3
    method_seg=getIWseg(img,nClusters=n)
    t=time.time()-t
    method_boun=find_boundaries(method_seg, mode='thick')
    bestMatchedGtIndex, fb, fop, ami, ari = getEvalScores(gtObj,gtBoun,method_boun,method_seg, evalPriority='ARI')
    showResultPlot(method_boun,method_seg,bestMatchedGtIndex, fb, fop, ami, ari, t, methodName='Iterated Watershed')