import os
import xml.etree.ElementTree as ET
import numpy as np
import pdb
LABEL_NAMES = np.array([
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor'])

def readxml(root,path):
    path=os.path.join(root,path)
    root=ET.parse(path).getroot()
    objs=root.findall("object")
    cls=list()
    boxes=[]
    tags=["xmin","ymin","xmax","ymax"]
    for obj in objs:
        name=obj.find("name")
        cls.append(np.argwhere(LABEL_NAMES==name.text).item()+1)
        bnd=obj.find("bndbox")
        box=[bnd.find(tag).text for tag in tags]
        boxes.append(box)
    return boxes,cls

def main(xmlDir,input,output):
    outPath=os.path.join(output)
    F=open(input,"r")
    filelist=F.readlines()
    np.random.shuffle(filelist)
    xmllist=["{}.xml".format(item.strip("\n")) for item in filelist]
    F.close()
    F1=open(outPath,"w")
    for index,xmlPath in enumerate(xmllist):
        boxes,cls=readxml(xmlDir,xmlPath)
        boxstr=[" ".join(item) for item in boxes]
        length=len(cls)
        F1.write("{} {}".format(xmlPath.replace("xml","jpg"),length))
        for i in range(length):
            F1.write(" {} {}".format(cls[i],boxstr[i]))
        F1.write("\n")
    F1.close()

if __name__=="__main__":
    input="/home/huangeryu/VOC/VOCdevkit/VOC2007/ImageSets/Main/test.txt"
    xmlDir="/home/huangeryu/VOC/VOCdevkit/VOC2007/Annotations"
    out="/home/huangeryu/Desktop/image/test.txt"
    main(xmlDir,input,out)