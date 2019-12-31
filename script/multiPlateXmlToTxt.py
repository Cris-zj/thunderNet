import numpy as np
import xml.etree.ElementTree as ET
import os
import pdb
import shutil

class multiData:
    def __init__(self,root):
        self.root=os.path.join(root)
        fileList=os.listdir(self.root)
        self.fileName=[item.split(".")[0] for item in fileList if os.path.isfile(os.path.join(self.root,item)) and item.endswith(".xml")]
        np.random.shuffle(self.fileName)
        # print(self.fileName)
    def __len__(self):
        return len(self.fileName)

    def __getitem__(self,index):
        xmlPath=os.path.join(self.root,"{}.xml".format(self.fileName[index]))
        return readXML(xmlPath)[:,0::2,:]
        
def main():
    root=os.path.join("/home/huangeryu/Desktop/plateLabel")
    data=multiData(root)
    DestPath=os.path.join(root,"multilabel.txt")
    with open(DestPath,"w+") as F:
        for index,rect in enumerate(data):
            if rect is None:
                print("{} broken!".format(data.fileName[index]))
                continue
            length=len(rect)
            label=rect.reshape(-1).tolist()
            F.write("{}.jpg {}".format(data.fileName[index],length))
            for i in range(length):
                F.write(" {} {} {} {} {}".format(0,label[i*4+1],label[i*4],label[i*4+3],label[i*4+2]))
            F.write("\n")
            # print("{}.jpg converted!".format(data.fileName[index]))
        print("convert done!")


def readXML(root):
    tree=ET.parse(root)
    objlist=tree.getroot().findall("object")
    labels=["xmin","ymin","xmax","ymax"]
    minIndex=0
    temp=list()
    if len(objlist)==1:
        itemlist=objlist[0].findall("item")
        for index,item in enumerate(itemlist):
            name=item.find("name").text
            if(name=="cn1"):
                minIndex=index
            bnd=item.find("bndbox")
            for label in labels:
                temp.append(int(bnd.find(label).text))
        if minIndex!=0:
            for i in range(4):
                buffer=temp[minIndex*4+i]
                temp[minIndex*4+i]=temp[i]
                temp[i]=buffer
    elif len(objlist)>=4:
        if(len(objlist)%4!=0):
            print(root)
            return None
        for index,item in enumerate(objlist):
            name=item.find("name").text
            if(name=="cn1" or name=="cn21" or name=="cn31" or name=="cn41"):
                minIndex=index
            bnd=item.find("bndbox")
            for label in labels:
                temp.append(int(bnd.find(label).text))
            if(index%4==3 and minIndex%4!=0):
                for i in range(4):
                    buffer=temp[minIndex*4+i]
                    temp[minIndex*4+i]=temp[index//4*4*4+i]
                    temp[index//4*4*4+i]=buffer
                minIndex=0
    temp=np.array(temp)
    temp=temp.reshape((-1,4,4))
    rects=np.zeros((len(temp),4,2),dtype=np.int)
    for index,item in enumerate(temp):
        assert(item.shape==(4,4))
        t=(item[:,0:2]+item[:,2:4])/2
        center=np.asarray(t,dtype=np.int)
        rects[index][0]=(item[0][1],item[0][0])
        for i in range(1,4):
            inds=[1,2,3]
            inds.remove(i)
            if(center[i][0]-center[0][0]==0): continue
            k=float(center[i][1]-center[0][1])/(center[i][0]-center[0][0])
            y1=k*(center[inds[0]][0]-center[0][0])+center[0][1]-center[inds[0]][1]
            y2=k*(center[inds[1]][0]-center[0][0])+center[0][1]-center[inds[1]][1]
            if(y1*y2<0):
                rects[index][2]=(item[i][3],item[i][2])  
                if(y1<0):
                    rects[index][3]=(item[inds[1]][3],item[inds[1]][0])
                    rects[index][1]=(item[inds[0]][1],item[inds[0]][2])
                else:
                    rects[index][1]=(item[inds[1]][3],item[inds[1]][0])
                    rects[index][3]=(item[inds[0]][1],item[inds[0]][2])
                break
    return rects

def replace(root):
    root=os.path.join(root)
    fileList=os.listdir(root)
    fileName=[item for item in fileList if os.path.isfile(os.path.join(root,item))]
    for item in fileName:
        src=os.path.join(root,item)
        destName=item.replace(" ","").replace(".JPG",".jpg")
        dest=os.path.join(root,destName)
        shutil.move(src,dest)
    print("done!")

if __name__=="__main__":
    main()
    # root="/home/huangeryu/Desktop/plateLabel"
    # replace(root)
