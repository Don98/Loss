import numpy as np
import os
import matplotlib.pyplot as plt
from color import cnames
from ap1 import epoch_ap
from ap1 import deal_table
import Don

all_color = list(cnames.keys())
# exit()
def get_cls_loss(cls_loss,pos2):
    # return all_loss sum , [classification , No.bbox , bce ,index.sum()]
    start = cls_loss[0].index("(")
    end   = cls_loss[0].index(", ")
    clss_loss_num = float(cls_loss[0][start+1:end].strip())
    pos = []
    new_cls_loss = []
    for i in range(len(cls_loss[1:])):
        if cls_loss[i+1].strip()[0] == "[":
            pos.append(i+1)
    pos.append(pos2-1)
    for i in range(len(pos) - 1):
        new_cls_loss.append("".join(cls_loss[pos[i]:pos[i+1]]))
        
    for i in range(len(new_cls_loss)):
        new_cls_loss[i] = new_cls_loss[i].replace(", ",",").split(" ")
        # print(new_cls_loss[i])
        new_cls_loss[i][0] = [float(j) for j in new_cls_loss[i][0][1:-1].split(",")]
        new_cls_loss[i][1] = int(new_cls_loss[i][1])
        # print(new_cls_loss[i][2])
        end = new_cls_loss[i][2].index("]")
        new_cls_loss[i][2] = [float(j) for j in new_cls_loss[i][2][1:end].split(",")]
        new_cls_loss[i][3] = float(new_cls_loss[i][3])
        
    return clss_loss_num,new_cls_loss

def get_reg_loss(reg_loss):
    reg_loss_num = float(reg_loss[0])
    new_reg_loss = []
    for i in reg_loss[1:]:
        tmp = i.replace(", ",",").split(" ")
        num = int(tmp[0])
        end = tmp[1].index("]")
        tmp = [float(j) for j in tmp[1][1:end].split(",")]
        new_reg_loss.append([num,tmp])
    return reg_loss_num,new_reg_loss

def read_file(filename):
    with open(filename,"r") as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    pos1 = 0
    pos2 = 0
    for i in range(len(data)):
        if data[i] == "="*50:
            pos1 = i
        if data[i] == "-"*50:
            pos2 = i
    if pos1 == 0:
        return [],0.0,[],0.0,[],0.0
    bbox     = data[:pos1]
    # print(bbox)
    bbox     = [[float(j) for j in i.replace(" ","")[1:-1].split(",")] for i in bbox]
    cls_loss = data[pos1+1:pos2]
    clss_loss_num, cls_loss = get_cls_loss(cls_loss,pos2)
    reg_loss_num , reg_loss = get_reg_loss(data[pos2+1:-1])
    all_loss = float(data[-1])
    return bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss

def deal_bbox(bbox):
    new_bbox = []
    for i in bbox:
        size = np.abs(i[2] - i[0]) * np.abs(i[3] - i[1])
        if size <= 1024:
            new_bbox.append([0,i[4]])
        elif size <= 9216:
            new_bbox.append([1,i[4]])
        else:
            new_bbox.append([2,i[4]])
    return np.array(new_bbox)   
        
def to_print(bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss):
    print("bbox".center(50,"-"))
    print(bbox)
    print("clss_loss_num".center(50,"-"))
    print(clss_loss_num)
    # print("cls_loss".center(50,"-"))
    # print(cls_loss)
    print("reg_loss_num".center(50,"-"))
    print(reg_loss_num)
    # print("reg_loss".center(50,"-"))
    # print(reg_loss)
    print("all_loss".center(50,"-"))
    print(all_loss)
def loadCats():
    file = open('classes.txt', 'r') 
    js = file.read()
    a = js[1:-1].split(", ")
    result = []
    for i in a:
        i = i.split(": ")
        result.append(i[1][1:-1])
    file.close()
    return result

def deal_data(path,epochs,filename_part,classes):
    f = open(path + ".csv","w")
    f1 = open(path + "_small.csv","w")
    f2 = open(path + "_middle.csv","w")
    f3 = open(path + "_large.csv","w")
    for i in range(epochs):
        clas_loss    = np.zeros((3,20))
        regr_loss    = np.zeros((3,20))
        num_class    = np.zeros((3,20))
        for j in range(2501):
        # for j in range(30):
            # print(j)
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(i) + os.sep + filename_part + str(j) + ".txt")
            # bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(9) + os.sep + filename_part + str(j) + ".txt")
            if bbox == [] and clss_loss_num == 0.0:
                continue
            bbox = deal_bbox(bbox)
            for k in cls_loss:
                clas_loss[int(bbox[k[1]][0])][int(bbox[k[1]][1])] += k[2][int(bbox[k[1]][1])]
            for k in reg_loss:
                regr_loss[int(bbox[k[0]][0])][int(bbox[k[0]][1])] += np.sum(k[1])
        # print(clas_loss.sum(axis=1))
        # np.savetxt(path + str(i) + "cls.txt",clas_loss)
        # np.savetxt(path + str(i) + "reg.txt",regr_loss)
        # print(clas_loss)
        # print(regr_loss)
        if i == 0:
            f.write(" , ,"+", , ,".join(classes) + "\n")
            f.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
            f1.write(" , ,"+", , ,".join(classes) + "\n")
            f1.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
            f2.write(" , ,"+", , ,".join(classes) + "\n")
            f2.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
            f3.write(" , ,"+", , ,".join(classes) + "\n")
            f3.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
        scale_name = ["Small","Middle","Large","ALL"]
        for i in range(clas_loss.shape[0]):
            f.write(scale_name[i] + ",")
            for j in range(clas_loss.shape[1]):
                f.write(str(clas_loss[i][j]) + "," + str(regr_loss[i][j]) + "," + str(clas_loss[i][j]+regr_loss[i][j]) + ",")
                if i == 0:
                    if j == 0:
                        f1.write(scale_name[i] + ",")
                    f1.write(str(clas_loss[i][j]) + "," + str(regr_loss[i][j]) + "," + str(clas_loss[i][j]+regr_loss[i][j]) + ",")
                if i == 1:
                    if j == 0:
                        f2.write(scale_name[i] + ",")
                    f2.write(str(clas_loss[i][j]) + "," + str(regr_loss[i][j]) + "," + str(clas_loss[i][j]+regr_loss[i][j]) + ",")
                if i == 2:
                    if j == 0:
                        f3.write(scale_name[i] + ",")
                    f3.write(str(clas_loss[i][j]) + "," + str(regr_loss[i][j]) + "," + str(clas_loss[i][j]+regr_loss[i][j]) + ",")
            f.write("\n")
            if i == 2:
                f.write(scale_name[-1] + ",")
                for j in range(clas_loss.shape[1]):
                    f.write(str(clas_loss[0][j]+clas_loss[1][j]+clas_loss[2][j]) + "," + str(regr_loss[0][j]+regr_loss[1][j]+regr_loss[2][j]) + "," + str(clas_loss[0][j]+clas_loss[1][j]+clas_loss[2][j] + regr_loss[0][j]+regr_loss[1][j]+regr_loss[2][j]) + ",")
                f.write("\n")
        f1.write("\n")
        f2.write("\n")
        f3.write("\n")
        
    f.close()
    f1.close()
    f2.close()
    f3.close()
    
def read_data_from_file():
    all_losses = []
    clas_losses = []
    regr_losses = []
    for i in range(17):
        clas_loss = np.loadtxt(path + str(i) + "cls.txt",delimiter=' ')
        regr_loss = np.loadtxt(path + str(i) + "reg.txt",delimiter=' ')
        clas_losses.append(clas_loss)
        regr_losses.append(regr_loss)
        all_losses.append(clas_loss + regr_loss)
    return all_losses,clas_losses,regr_losses
    
def draw_all_loss(epochs,all_losses):
    x = [i for i in range(epochs)]
    all_losses = [i.sum() for i in all_losses]
    plt.plot(x,all_losses)    
    plt.xlabel("Epoch")
    plt.ylabel("All losses")
    plt.legend()
    plt.show()
    
def draw_20_classes(classes,all_losses):
    all_classes_data = [[] for i in range(len(classes))]
    x = [i for i in range(epochs)]
    for i in range(epochs):
        all_losses[i] = all_losses[i].sum(axis = 0)
        for j in range(len(classes)):
            all_classes_data[j].append(all_losses[i][j])
    for i in range(len(classes)):
        plt.plot(x,all_classes_data[i],label=classes[i],color=all_color[i])
    tmp = [i.sum() / len(classes) for i in all_losses]
    plt.plot(x,tmp,label="Mean")    
    plt.xlabel("Epoch")
    plt.ylabel("All losses")
    plt.title("20 classes losses")
    plt.legend()
    plt.show()
    
def draw_sclae_classes(classes,all_losses,scale):
    all_classes_data = [[] for i in range(len(classes))]
    x = [i for i in range(epochs)]
    for i in range(epochs):
        # print(all_losses[i])
        all_losses[i] = all_losses[i][scale]
        for j in range(len(classes)):
            all_classes_data[j].append(all_losses[i][j])
    for i in range(len(classes)):
        plt.plot(x,all_classes_data[i],label=classes[i],color=all_color[i])
    tmp = [i[scale] / len(classes) for i in all_losses]
    plt.plot(x,tmp,label="Mean")    
    plt.xlabel("Epoch")
    plt.ylabel("All losses")
    scale_name = ["Small","Middle","Large"]
    plt.title(scale_name[scale] + " losses")
    plt.legend()
    plt.show()
    
def deal_data1(classes):
    f= open("VOC_train_num.csv","r")
    data = f.readlines()[1:-3]
    f.close()
    all_class_num = [[] for i in range(len(classes))]
    pos = 0
    for i in range(len(data)):
        if (i+1) % 3 == 0:
            continue
        data[i] = data[i].strip().split(",")
        if data[i][0] != "":
            pos = classes.index(data[i][0])
        data[i] = [float(j) for j in data[i][2:]]
        all_class_num[pos].append(data[i])
    f = open(path + ".csv","w")
    f1 = open(path + "_small.csv","w")
    f2 = open(path + "_middle.csv","w")
    f3 = open(path + "_large.csv","w")
    for i in range(epochs):
        clas_loss    = np.zeros((3,20))
        regr_loss    = np.zeros((3,20))
        num_class    = np.zeros((3,20))
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(i) + os.sep + filename_part + str(j) + ".txt")
            if bbox == [] and clss_loss_num == 0.0:
                continue
            bbox = deal_bbox(bbox)
            for k in cls_loss:
                clas_loss[int(bbox[k[1]][0])][int(bbox[k[1]][1])] += k[2][int(bbox[k[1]][1])]
            for k in reg_loss:
                regr_loss[int(bbox[k[0]][0])][int(bbox[k[0]][1])] += np.sum(k[1])
        if i == 0:
            f.write(" , ,"+", , ,".join(classes) + "\n")
            f.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
            f1.write(" , ,"+", , ,".join(classes) + "\n")
            f1.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
            f2.write(" , ,"+", , ,".join(classes) + "\n")
            f2.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
            f3.write(" , ,"+", , ,".join(classes) + "\n")
            f3.write(" ,"+",".join(["class,regr,all"] * len(classes)) + "\n")
        scale_name = ["Small_per_bbox","Small_per_picture","Middle_per_bbox","Middle_per_picture","Large_per_bbox","Large_per_picture","ALL_per_bbox","ALL_per_picture"]
        for i in range(clas_loss.shape[0]):
            f.write(scale_name[i*2] + ",")
            for j in range(clas_loss.shape[1]):
                f.write(str(clas_loss[i][j] / all_class_num[j][0][i]) + "," + str(regr_loss[i][j]/ all_class_num[j][0][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j])/ all_class_num[j][0][i]) + ",")
            f.write("\n")
            f.write(scale_name[i*2+1] + ",")
            for j in range(clas_loss.shape[1]):
                f.write(str(clas_loss[i][j] / all_class_num[j][1][i]) + "," + str(regr_loss[i][j]/ all_class_num[j][1][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j])/ all_class_num[j][1][i]) + ",")
            f.write("\n")
            if i == 2:
                for j in range(clas_loss.shape[1]):
                    if j == 0:
                        f.write(scale_name[-2] + ",")
                    f.write(str((clas_loss[0][j]+clas_loss[1][j]+clas_loss[2][j])/ all_class_num[j][0][i]) + "," + str((regr_loss[0][j]+regr_loss[1][j]+regr_loss[2][j])/ all_class_num[j][0][i]) + "," + str((clas_loss[0][j]+clas_loss[1][j]+clas_loss[2][j] + regr_loss[0][j]+regr_loss[1][j]+regr_loss[2][j])/ all_class_num[j][0][i]) + ",")
                for j in range(clas_loss.shape[1]):
                    if j == 0:
                        f.write(scale_name[-1] + ",")
                    f.write(str((clas_loss[0][j]+clas_loss[1][j]+clas_loss[2][j])/ all_class_num[j][1][i]) + "," + str((regr_loss[0][j]+regr_loss[1][j]+regr_loss[2][j])/ all_class_num[j][1][i]) + "," + str((clas_loss[0][j]+clas_loss[1][j]+clas_loss[2][j] + regr_loss[0][j]+regr_loss[1][j]+regr_loss[2][j])/ all_class_num[j][1][i]) + ",")
                f.write("\n")
        for i in range(clas_loss.shape[0]):
            for j in range(clas_loss.shape[1]):
                if i == 0:
                    if j == 0:
                        f1.write(scale_name[i*2] + ",")
                    f1.write(str(clas_loss[i][j] / all_class_num[j][0][i]) + "," + str(regr_loss[i][j] / all_class_num[j][0][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j]) / all_class_num[j][0][i]) + ",")
                if i == 1:
                    if j == 0:
                        f2.write(scale_name[i*2] + ",")    
                    f2.write(str(clas_loss[i][j] / all_class_num[j][0][i]) + "," + str(regr_loss[i][j] / all_class_num[j][0][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j]) / all_class_num[j][0][i]) + ",")
                if i == 2:
                    if j == 0:
                        f3.write(scale_name[i*2] + ",")
                    f3.write(str(clas_loss[i][j] / all_class_num[j][0][i]) + "," + str(regr_loss[i][j] / all_class_num[j][0][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j]) / all_class_num[j][0][i]) + ",")
            if i == 0:
                f1.write("\n")
            if i == 1:
                f2.write("\n")
            if i == 2:
                f3.write("\n")
            for j in range(clas_loss.shape[1]):
                if i == 0:
                    if j == 0:
                        f1.write(scale_name[i*2+1] + ",")
                    f1.write(str(clas_loss[i][j] / all_class_num[j][1][i]) + "," + str(regr_loss[i][j] / all_class_num[j][1][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j]) / all_class_num[j][1][i]) + ",")
                if i == 1:
                    if j == 0:
                        f2.write(scale_name[i*2+1] + ",")    
                    f2.write(str(clas_loss[i][j] / all_class_num[j][1][i]) + "," + str(regr_loss[i][j] / all_class_num[j][1][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j]) / all_class_num[j][1][i]) + ",")
                if i == 2:
                    if j == 0:
                        f3.write(scale_name[i*2+1] + ",")
                    f3.write(str(clas_loss[i][j] / all_class_num[j][1][i]) + "," + str(regr_loss[i][j] / all_class_num[j][1][i]) + "," + str((clas_loss[i][j]+regr_loss[i][j]) / all_class_num[j][1][i]) + ",")
        f1.write("\n")
        f2.write("\n")
        f3.write("\n")
        
    f.close()
    f1.close()
    f2.close()
    f3.close()
    
def draw_20_ap(epochs,classes):
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    Y = [[] for i in range(len(classes) + 1)]
    the_name = []
    for i in epoch_ap:
        L,name = deal_table(i)
        the_name = name
        for j in range(len(the_name)):
            Y[j].append(L[0][j])
    for i in range(len(the_name)):
        plt.plot(x,Y[i],label=the_name[i],color=all_color[i])
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    plt.title("20 classes AP")
    plt.legend()
    plt.show()
    
def draw_scale_ap(epochs,classes,scale):
    x = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    Y = [[] for i in range(len(classes) + 1)]
    the_name = []
    for i in epoch_ap:
        L,name = deal_table(i)
        the_name = name
        for j in range(len(the_name)):
            Y[j].append(L[scale][j])
    # print(len(x))
    # print(len(Y[0]))
    for i in range(len(the_name)):
        plt.plot(x,Y[i],label=the_name[i],color=all_color[i])
    plt.xlabel("Epoch")
    plt.ylabel("AP")
    scale_name = ["Small","Middle","Large"]
    plt.title(scale_name[scale-1] + " AP")
    plt.legend()
    plt.show()
        
def draw_each_class_ap(epochs,all_losses,classes,the_class):
    the_color = ["r","g","b","black"]
    Y = [[] for i in range(8)]
    for i in range(epochs):
        tmp = 0
        for j in range(3):
            Y[j].append(all_losses[i][j][the_class])
            tmp += all_losses[i][j][the_class]
        Y[3].append(tmp)
    x = [i for i in range(epochs)]
    x1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    the_name = []
    for i in epoch_ap:
        L,name = deal_table(i)
        the_name = name
        for j in range(4):
            Y[4 + j].append(L[(j + 1) % 4][the_class])    
    # for i in range(4):
        # tmp = Y[i][0]
        # for j in range(len(Y[i])):
            # if tmp == 0.0:
                # Y[i][j] = 0
            # else:
                # Y[i][j] = 1 - (tmp - Y[i][j]) / tmp
    # for i in range(4):
        # print(Y[i])
    fig = plt.figure()

    scale_name = ["Small","Middle","Large","ALL"]
    ax1 = fig.add_subplot(111)
    for i in range(1):
    # for i in range(4):
        ax1.plot(x, Y[i],label=scale_name[i] + "_loss",color = the_color[i],linestyle="--")
    ax1.set_ylabel('the rest of loss')
    ax1.set_title(classes[the_class])

    ax2 = ax1.twinx()  # this is the important function
    for i in range(4,8):
        ax2.plot(x1, Y[i],label=scale_name[i-4] + "_ap",color = the_color[i-4],linestyle="-")
    ax2.set_ylabel('the AP')
    ax2.set_xlabel('Epoch')
    plt.legend()
    # print(classes[the_class])
    # plt.savefig("pic/classes/" + classes[the_class] + ".png")
    # plt.savefig("pic/classes_change/" + classes[the_class] + ".png")
    plt.savefig("pic/small_losses/" + classes[the_class] + ".png")
    # plt.show()
    
def all_sample(epochs,classes):
    f = open("16sample_loss.csv","w")
    f.write("属于Epoch的样本,loss,类别,属于尺度,bbox,each_loss\n")
    each_iter = []
    for epoch in range(epochs):
        for j in range(2501):
        # for j in range(20):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(epoch) + os.sep + filename_part + str(j) + ".txt")
            result = np.zeros((len(bbox),1))
            scale = []
            each_classes = []
            for k in bbox:
                size = np.abs(k[2]-k[0]) * np.abs(k[3] - k[1])
                if size <= 1024:
                    scale.append("S")
                elif size <= 9216:
                    scale.append("M")
                else:
                    scale.append("L")
                each_classes.append(classes[int(k[4])])
            
            for k in range(len(cls_loss)):
                result[cls_loss[k][1]] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
            write_content = [",".join([str(epoch),str(result.sum()),each_classes[0],scale[0],str(bbox[0]).replace(","," "),str(result[0][0])]) + "\n"]
            for i in range(1,len(bbox)):
                write_content[0] += ",".join(["","",each_classes[i],scale[i],str(bbox[i]).replace(","," "),str(result[i][0])]) + "\n"
            write_content.append(result.sum())
            each_iter.append(write_content)
    each_iter.sort(key = lambda x : x[1])
    for i in each_iter[::-1]:
        f.write(i[0])
    f.close()
def all_bbox(epochs,classes):
    f = open("16bbox_loss.csv","w")
    f.write("属于Epoch的样本,each_loss,类别,属于尺度,bbox\n")
    each_iter = []
    for epoch in range(epochs):
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(epoch) + os.sep + filename_part + str(j) + ".txt")
            result = np.zeros((len(bbox),1))
            scale = []
            each_classes = []
            for k in bbox:
                size = np.abs(k[2]-k[0]) * np.abs(k[3] - k[1])
                if size <= 1024:
                    scale.append("S")
                elif size <= 9216:
                    scale.append("M")
                else:
                    scale.append("L")
                each_classes.append(classes[int(k[4])])
            
            for k in range(len(cls_loss)):
                result[cls_loss[k][1]] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
            # write_content = [",".join([str(epoch),str(result.sum()),each_classes[0],scale[0],str(bbox[0]).replace(","," "),str(result[0][0])]) + "\n"]
            
            for i in range(len(bbox)):
                write_content = [""]
                write_content[0] += ",".join([str(epoch),str(result[i][0]),each_classes[i],scale[i],str(bbox[i]).replace(","," ")]) + "\n"
                write_content.append(result[i][0])
                each_iter.append(write_content)
    each_iter.sort(key = lambda x : x[1])
    for i in each_iter[::-1]:
        f.write(i[0])
    f.close()
    
def all_bbox_size(epochs,classes):
    f = open("16bbox_size_loss.csv","w")
    f.write("属于Epoch的样本,each_loss,类别,属于尺度,bbox,size\n")
    each_iter = []
    for epoch in range(epochs):
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(epoch) + os.sep + filename_part + str(j) + ".txt")
            result = np.zeros((len(bbox),1))
            scale = []
            each_classes = []
            all_size = []
            for k in bbox:
                size = np.abs(k[2]-k[0]) * np.abs(k[3] - k[1])
                all_size.append(size)
                if size <= 1024:
                    scale.append("S")
                elif size <= 9216:
                    scale.append("M")
                else:
                    scale.append("L")
                each_classes.append(classes[int(k[4])])
            
            for k in range(len(cls_loss)):
                result[cls_loss[k][1]] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
            # write_content = [",".join([str(epoch),str(result.sum()),each_classes[0],scale[0],str(bbox[0]).replace(","," "),str(result[0][0])]) + "\n"]
            
            for i in range(len(bbox)):
                write_content = [""]
                write_content[0] += ",".join([str(epoch),str(result[i][0]),each_classes[i],scale[i],str(bbox[i]).replace(","," "),str(all_size[i])]) + "\n"
                write_content.append(all_size[i])
                each_iter.append(write_content)
    each_iter.sort(key = lambda x : x[1])
    for i in each_iter[::-1]:
        f.write(i[0])
    f.close()
    
def get_classification(bbox,cls_loss):
    classification = []
    for i in cls_loss[1:]:
        i = i.replace(", ",",").split(" ")
        i[1] = int(i[1])
        i[0] = i[0][1:-1].split(",")
        classification.append(float(i[0][int(bbox[i[1]][4])]))
    return classification
def read_file1(filename):
    with open(filename,"r") as f:
        data = f.readlines()
    data = [i.strip() for i in data]
    pos1 = 0
    pos2 = 0
    for i in range(len(data)):
        if data[i] == "="*50:
            pos1 = i
        if data[i] == "-"*50:
            pos2 = i
    if pos1 == 0:
        return [],0.0,[],0.0,[],0.0
    bbox     = data[:pos1]
    bbox     = [[float(j) for j in i.replace(" ","")[1:-1].split(",")] for i in bbox]
    cls_loss = data[pos1+1:pos2]
    classification = get_classification(bbox,cls_loss)
    clss_loss_num, cls_loss = get_cls_loss(cls_loss,pos2)
    reg_loss_num , reg_loss = get_reg_loss(data[pos2+1:-1])
    all_loss = float(data[-1])
    return bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss,classification
def draw_clas_loss(epochs,classes):
    z = []
    for epoch in range(epochs):
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss,classification = read_file1(path + str(epoch) + os.sep + filename_part + str(1) + ".txt")
            for k in range(len(bbox)):
                pos = int(bbox[k][4])
                loss = cls_loss[k][2][pos] + sum(reg_loss[k][1])
                z.append((classification[k],loss))
    z.sort(key = lambda x : x[0])
    x = []
    y = []
    print(len(z))
    for i in z:
        x.append(i[0])
        y.append(i[1])
    plt.plot(x,y)
    print(len(x),len(y))
    plt.xlabel("probability of ground truth class")
    plt.ylabel("loss")
    plt.title("classification-loss")
    plt.legend()
    plt.show()

def all_bbox_num(epochs,classes):
    f = open("16bbox_num_loss.csv","w")
    f.write("类别,尺度,样本数量,loss,平均\n")
    result = np.zeros((3,20))
    result_num = np.zeros((3,20))
    for epoch in range(epochs):
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(epoch) + os.sep + filename_part + str(j) + ".txt")
            scale = []
            each_classes = []
            for k in bbox:
                size = np.abs(k[2]-k[0]) * np.abs(k[3] - k[1])
                if size <= 1024:
                    scale.append(0)
                elif size <= 9216:
                    scale.append(1)
                else:
                    scale.append(2)
                each_classes.append(classes[int(k[4])])
                result_num[scale[-1]][int(k[4])] += 1
            for k in range(len(cls_loss)):
                result[scale[cls_loss[k][1]]][int(bbox[cls_loss[k][1]][4])] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
    num = 0
    scale_name = ["S","M","L"]
    for i in range(len(classes)):
        for j in range(3):
            if j == 1:
                the_num = 0
                if result_num[j][i] != 0:
                    the_num = result[j][i] / result_num[j][i]
                f.write(classes[num] + ",M" + "," + str(result_num[j][i]) + "," + str(result[j][i]) + "," + str(the_num) + "\n")
                num += 1
            else:
                the_num = 0
                if result_num[j][i] != 0:
                    the_num = result[j][i] / result_num[j][i]
                f.write("," + scale_name[j] + "," + str(result_num[j][i]) + "," + str(result[j][i]) + "," + str(the_num) + "\n")
    f.close()
def all_bbox_num_sort(epochs,classes):
    f = open("16bbox_num_nosort_loss.csv","w")
    # f = open("16bbox_num_sort_loss.csv","w")
    # f = open("16bbox_sort_loss.csv","w")
    f.write("类别,尺度,Bboxs数量,样本数量,loss,平均,剩余loss")
    for i in range(1,epochs):
        f.write(",Epoch" + str(i))
    f.write("\n")
    result = np.zeros((3,20))
    result_num = np.zeros((3,20))
    each_iter = []
    sample_num = np.zeros((3,20))
    start_loss = np.zeros((3,20))
    tmp_loss   = np.zeros((3,20))
    end_loss   = np.zeros((3,20))
    for epoch in range(epochs):
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(epoch) + os.sep + filename_part + str(j) + ".txt")
            scale = []
            each_classes = []
            part_sample_num = np.zeros((3,20))
            for k in bbox:
                size = np.abs(k[2]-k[0]) * np.abs(k[3] - k[1])
                if size <= 1024:
                    scale.append(0)
                elif size <= 9216:
                    scale.append(1)
                else:
                    scale.append(2)
                each_classes.append(classes[int(k[4])])
                result_num[scale[-1]][int(k[4])] += 1
                part_sample_num[scale[-1]][int(k[4])] = 1
            sample_num += part_sample_num
            for k in range(len(cls_loss)):
                result[scale[cls_loss[k][1]]][int(bbox[cls_loss[k][1]][4])] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
                if epoch == 0:
                    start_loss[scale[cls_loss[k][1]]][int(bbox[cls_loss[k][1]][4])] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
                if epoch == epochs - 1:
                    end_loss[scale[cls_loss[k][1]]][int(bbox[cls_loss[k][1]][4])] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
    # print(end_loss)
    # print(start_loss)
    the_pos = (start_loss == 0)
    end_loss = 1 - (start_loss - end_loss) / start_loss
    end_loss[the_pos] = 0.0
    scale_name = ["S","M","L"]
    for i in range(len(classes)):
        for j in range(3):
            the_num = 0
            if sample_num[j][i] != 0:
                the_num = result[j][i] / sample_num[j][i]
            write_content = [classes[i] + ","+ scale_name[j] + "," + str(result_num[j][i]) + ","+ str(sample_num[j][i]) + "," + str(result[j][i]) + "," + str(the_num) + "," + str(end_loss[j][i]),result_num[j][i],result[j][i],i,j]
            each_iter.append(write_content)
    Y = []
    the_name = []
    for i in epoch_ap:
        L,name = deal_table(i)
        the_name = name
        Y.append(L)
    # each_iter.sort(key = lambda x :x[1])
    # for i in each_iter[::-1]:
    for i in each_iter:
        f.write(i[0])
        for j in range(epochs - 1):
            f.write("," + str(Y[j][i[-1]+1][i[-2]]))
        f.write("\n")
    f.close()
def each_loss_num_ap(epochs,classes):
    # f = open("each_loss_num_ap.csv","w")
    f = open("each_loss_num_nosort_ap.csv","w")
    f.write("类别,尺度,loss")
    for i in range(epochs):
        f.write(",Epoch" + str(i))
    f.write("\n")
    result = np.zeros((3,20))
    result = [list(i) for i in list(result)]
    for i in range(3):
        for j in range(20):
            result[i][j] = [0.0 for i in range(epochs)]
    import copy
    # result_num = np.zeros((3,20))
    result_num = copy.deepcopy(result)
    each_iter = []
    for epoch in range(epochs):
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(epoch) + os.sep + filename_part + str(j) + ".txt")
            scale = []
            each_classes = []
            for k in bbox:
                size = np.abs(k[2]-k[0]) * np.abs(k[3] - k[1])
                if size <= 1024:
                    scale.append(0)
                elif size <= 9216:
                    scale.append(1)
                else:
                    scale.append(2)
                each_classes.append(classes[int(k[4])])
                result_num[scale[-1]][int(k[4])][epoch] += 1
            for k in range(len(cls_loss)):
                result[scale[cls_loss[k][1]]][int(bbox[cls_loss[k][1]][4])][epoch] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
    scale_name = ["S","M","L"]
    for i in range(len(classes)):
        for j in range(3):
            the_num = 0
            if sum(result_num[j][i]) != 0:
                the_num = sum(result[j][i]) / sum(result_num[j][i])
            write_content = [classes[i] + ","+ scale_name[j],sum(result_num[j][i]),result_num[j][i],result[j][i],i,j]
            each_iter.append(write_content)
    Y = []
    the_name = []
    for i in epoch_ap:
        L,name = deal_table(i)
        the_name = name
        Y.append(L)
    # each_iter.sort(key = lambda x :x[1])
    # for i in each_iter[::-1]:
    for i in each_iter:
        f.write(i[0])
        f.write(",AP:,")
        for j in range(epochs-1):
            f.write("," + str(Y[j][i[-1]+1][i[-2]]))
        f.write("\n")
        f.write(",,loss:")
        for j in range(epochs):
            f.write("," + str(i[-3][j]))
        f.write("\n")
        f.write(",,loss变化:")
        for j in range(epochs):
            if i[-3][0] == 0:
                f.write("," + str(0))
            else:
                f.write("," + str(1 - (i[-3][0] - i[-3][j]) / i[-3][0]))
        f.write("\n")
        f.write(",,mean:")
        for j in range(epochs):
            if i[-4][j] != 0:
                f.write("," + str(i[-3][j] / i[-4][j]))
            else:
                f.write("," + str(0))
        f.write("\n")
    f.close()

def draw_scale_loss(epochs,classes):
    result_num = np.zeros((3,20))
    each_iter = []
    for epoch in range(1):
        for j in range(2501):
            bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file(path + str(epoch) + os.sep + filename_part + str(j) + ".txt")
            all_size = []
            result = np.zeros((len(bbox),1))
            for k in bbox:
                size = np.abs(k[2]-k[0]) * np.abs(k[3] - k[1])
                all_size.append(size)
            for k in range(len(cls_loss)):
                result[cls_loss[k][1]] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
                # result[scale[cls_loss[k][1]]][int(bbox[cls_loss[k][1]][4])] += cls_loss[k][2][int(bbox[cls_loss[k][1]][4])] + sum(reg_loss[k][1])
            for k in range(result.shape[0]):
                each_iter.append((all_size[k],result[k]))
    each_iter.sort(key = lambda x :x[0])
    X = [25];Y = [each_iter[0]]
    num = 25
    the_num = 0
    for i in each_iter:
        if i[0] - num <= 10000:
            Y[the_num] += i[1]
        else:
            num += 25
            X.append(num + 25)
            Y.append(i[1])
            the_num += 1
    for i in each_iter:
        print(i)
    plt.plot(X,Y)
    plt.xlabel("the size of the object")
    plt.ylabel("loss")
    plt.title("scale-loss")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    epoch_loss = np.zeros(())
    epochs = 17
    path = "Epoch"
    filename_part = "record"
    classes = loadCats()
    # print(classes)
    
    # bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss = read_file("Epoch0\\record6.txt")
    # to_print(bbox,clss_loss_num,cls_loss,reg_loss_num,reg_loss,all_loss)
    # deal_data(path,epochs,filename_part,classes)
    all_losses,clas_losses,regr_losses = read_data_from_file()
    # deal_data1(classes)
    
    # draw_all_loss(epochs,all_losses)
    # draw_20_classes(classes,all_losses)
    # draw_sclae_classes(classes,all_losses,0)
    # draw_sclae_classes(classes,all_losses,1)
    # draw_sclae_classes(classes,all_losses,2)
    
    # draw_20_ap(epochs,classes)
    # draw_scale_ap(epochs,classes,1)
    # draw_scale_ap(epochs,classes,2)
    # draw_scale_ap(epochs,classes,3)
    # for i in range(len(classes)):
        # draw_each_class_ap(epochs,all_losses,classes,i)
        # draw_each_class_ap(epochs,all_losses,classes,5)
        # exit()
    # all_sample(epochs,classes)
    # all_bbox(epochs,classes)
    # all_bbox_size(epochs,classes)
    # draw_clas_loss(epochs,classes)
    # all_bbox_num(epochs,classes)
    # all_bbox_num_sort(epochs,classes)
    # each_loss_num_ap(epochs,classes)   
    draw_scale_loss(epochs,classes)
    Don.Mess()