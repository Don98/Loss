import os
import copy

def getLoadAnnIds(annot_path,imgIds):
    import re
    f= open(annot_path + imgIds,"r")
    data = f.read()
    objects = re.compile("<object>([\w\W]+?)</object>").findall(data)
    result = []
    width = re.compile("<width>([\w\W]+?)</width>").findall(data)[0].strip()
    height = re.compile("<height>([\w\W]+?)</height>").findall(data)[0].strip()
    for i in objects:
        name = re.compile("<name>([\w\W]+?)</name>").findall(i)[0].strip()
        bndbox = re.compile("<bndbox>([\w\W]+?)</bndbox>").findall(i)[0].strip()
        nums = [float(i) for i in re.compile("<[\w\W]+?>([\w\W]+?)</[\w\W]+?>").findall(bndbox)]
        nums.append(name)
        result.append(nums)
    f.close()
    return result,width,height
    
def to_stastics(path,all_files):
    all_object = []
    for i in all_files:
        objects = getLoadAnnIds(path,i)[0]
        for j in objects:
            all_object.append((int(j[3]) - int(j[1])) * (int(j[2]) - int(j[0])))
    all_object = sorted(all_object)
    pos = int(len(all_object)*0.7)
    # print(all_object[:int(len(all_object)*0.7)])
    print(all_object[pos])
    for i in range(pos,pos+100):
        if all_object[i] > 180*180:
            print(i / len(all_object))
            break
    
path = "E:/VOC2007/VOCdevkit/VOC2007/Annotations/"
# set_path = "E:/VOC2007/VOCdevkit/VOC2007/ImageSets/Layout/" + "train2" + ".txt"
set_path = "E:/VOC2007/VOCdevkit/VOC2007/ImageSets/Layout/" + "val2" + ".txt"
# set_path = "E:/VOC2007/VOCdevkit/VOC2007/ImageSets/Layout/" + "trainval2" + ".txt"
f = open(set_path,"r")
all_files = [i.strip()  + ".xml"for i in f.readlines()]
f.close()

to_stastics(path,all_files)
# exit()
# all_files = os.listdir(path)
classes = []
classes = ["car","person","horse","bicycle","aeroplane","train","diningtable","dog","chair","cat","bird","boat","pottedplant","tvmonitor","sofa","motorbike","bottle","bus","sheep","cow"]

all_classes = []
for i in range(len(classes)):
    all_classes.append([0,0,0,0,0,0])
all_pic = len(all_files)
sma = 0
mid = 0
lar = 0
all_bboxex = 0
for i in all_files:
    objects = getLoadAnnIds(path,i)[0]
    tmp_classes = copy.deepcopy(all_classes)

    scale_flag = [0,0,0]
    for j in objects:
        all_bboxex += 1
        pos = -1
        if j[4] in classes:
            pos = classes.index(j[4])
        else:
            classes.append(j[4])
            all_classes.append([0,0,0,0,0,0])
            tmp_classes.append([0,0,0,0,0,0])
        size = (int(j[3]) - int(j[1])) * (int(j[2]) - int(j[0]))
        if size <= 1024:
            all_classes[pos][0] += 1
            tmp_classes[pos][1] += 1
            scale_flag[0] = 1
        elif size <= 9216:
            all_classes[pos][2] += 1
            tmp_classes[pos][3] += 1
            scale_flag[1] = 1
        else:
            all_classes[pos][4] += 1
            tmp_classes[pos][5] += 1
            scale_flag[2] = 1
    sma += scale_flag[0]
    mid += scale_flag[1]
    lar += scale_flag[2]
    for j in range(len(tmp_classes)):
        for k in range(1,6,2):
            if all_classes[j][k] < tmp_classes[j][k]:
                all_classes[j][k] += 1
# print(all_bboxex)
# print(all_pic)
scale_num = [0,0,0]
for i in range(len(classes)):
    for k in range(6):
        if k == 0:
            scale_num[0] += all_classes[i][k]
        elif k == 2:
            scale_num[1] += all_classes[i][k]
        elif k == 4:
            scale_num[2] += all_classes[i][k]
        if k % 2 == 0:
            # all_classes[i][k] = round(all_classes[i][k] / all_bboxex * 100,3)
            all_classes[i][k] = all_classes[i][k]
        else:
            # all_classes[i][k] = round(all_classes[i][k] / all_pic * 100,3)
            all_classes[i][k] = all_classes[i][k]


# with open("VOC_result.csv","w") as f:
    # f.write("class,,Sma,Mid,Lar\n")
    # for i in range(len(classes)):
        # f.write(classes[i] + ",")
        # L = ",".join([str(j) for j in all_classes[i][0::2]])
        # f.write("Ratio of total boxes%," + L + "\n")
        # L = ",".join([str(j) for j in all_classes[i][1::2]])
        # f.write(" ,Ratio of images included%, " + L + "\n")
    # f.write("All,Ratio of total boxes%," + str(round(scale_num[0]/all_bboxex * 100,3)) + "," + str(round(scale_num[1]/all_bboxex * 100,3)) + "," + str(round(scale_num[2]/all_bboxex * 100,3)) + "\n")
    # f.write(" ,Ratio of images included%," + str(round(sma/all_pic * 100,3)) + "," + str(round(mid/all_pic * 100,3)) + "," + str(round(lar/all_pic * 100,3)))
        
# with open("VOC_test_num.csv","w") as f:
# with open("VOC_train_num.csv","w") as f:
with open("VOC_val_num.csv","w") as f:
    f.write("class,,Sma,Mid,Lar\n")
    for i in range(len(classes)):
        f.write(classes[i] + ",")
        L = ",".join([str(j) for j in all_classes[i][0::2]])
        f.write("Total boxes," + L + "\n")
        L = ",".join([str(j) for j in all_classes[i][1::2]])
        f.write(" ,Images included, " + L + "\n")
        L = []
        for j in range(1,6,2):
            if(all_classes[i][j] != 0):
                L.append(str(all_classes[i][j - 1] / all_classes[i][j]))
            else:
                L.append(str(0))
        L = ",".join(L)
        f.write(" ,boxes / image," + L + "\n")
    f.write("All,Total boxes%," + str(scale_num[0]) + "," + str(scale_num[1]) + "," + str(scale_num[2]) + "\n")
    f.write(" ,Images included%," + str(sma) + "," + str(mid) + "," + str(lar) + "\n")
    f.write(" ,boxes / image," + str(scale_num[0] / sma) + "," + str(scale_num[1] / mid) + "," + str(scale_num[2] / lar ) + "\n")