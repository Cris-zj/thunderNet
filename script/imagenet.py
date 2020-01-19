import os
import numpy as np
import pdb

def uncompresstrain(root):
    root=os.path.join(root)
    tarlist=[item.split(".")[0] for item in os.listdir(root) if item.endswith("tar")]
    # for item in tarlist:
    #     os.system("tar -xvf {} -C {}".format(os.path.join(root,"{}.tar".format(item)),root))
    print("uncompress done!")
    label=os.path.join(root,"label.txt")
    with open(label,"w") as F:
        labellist="\n".join(tarlist)
        F.writelines(labellist)
    jpeglist=[item for item in os.listdir(root) if item.endswith(".JPEG")]
    np.random.shuffle(jpeglist)
    train=os.path.join(root,"train.txt")
    with open(train,"w") as F:
        for item in jpeglist:
            # pdb.set_trace()
            name=item.split("_")[0]
            index=tarlist.index(name)
            F.write("{} {}\n".format(item,index))
    print("train.txt written!")

def genval(root,labelmap,caffelabel):
    '''
    root: 验证集的目录,字符串类型
    labelmap: caffe版本的标签转本项目中采用的标签(两个之间只是顺序不同)，字典类型{int,int}
    caffelabel: caffe版本的验证集标签,字符串类型
    '''
    root=os.path.join(root)
    val=os.path.join(root,"val.txt")
    with open(caffelabel,"r") as F,open(val,"w") as W:
        lines=F.readlines()
        for line in lines:
            line=line.strip("\n").split(" ")
            W.write("{} {}\n".format(line[0],labelmap[int(line[1])]))
    print("genval done!")

def genmap(label,caffelabel):
    '''
    label: 自定义的标签,字符串类型
    caffelabel: imagenet caffe版本的synsets.txt路径，字符串类型
    '''
    idict=dict()
    with open(label,"r") as F:
        lines=F.readlines()
        for index,line in enumerate(lines):
            line=line.strip("\n")
            idict[line]=index
    caffedict=dict()
    with open(caffelabel,"r") as F:
        lines=F.readlines()
        for index,line in enumerate(lines):
            line=line.strip("\n")
            caffedict[line]=index
    ret=dict()
    for key,value in caffedict.items():
        ret[value]=idict[key]
    return ret

def main():
    # uncompresstrain("/home/huangeryu/Data/imagenet/train")
    caffemap=genmap("/home/huangeryu/Data/imagenet/train/label.txt","/home/huangeryu/Data/caffe_ilsvrc12/synsets.txt")
    genval("/home/huangeryu/Data/imagenet/val",caffemap,"/home/huangeryu/Data/caffe_ilsvrc12/val.txt")
    genval("/home/huangeryu/Data/imagenet/test",caffemap,"/home/huangeryu/Data/caffe_ilsvrc12/test.txt")

if __name__=="__main__":
    main()