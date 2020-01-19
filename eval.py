import visdom
import numpy as np

class confusionMatrix(object):
    def __init__(self,nums):
        self.cfsm=np.zeros((nums,nums),dtype=np.int)
    
    def setvalue(self,predict,gt_index):
        self.cfsm[gt_index,predict]+=1

    def show(self):
        temp=self.cfsm.sum(axis=1).reshape(-1,1)
        pmatrix=self.cfsm/temp
        confusionImg=np.asarray(pmatrix,dtype=np.float32)
        return confusionImg