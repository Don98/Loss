import random
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import axes
import numpy as np
import pandas as pd
import seaborn as sns
    
def draw():
    f = open("file/VOC2012/VOC2012.txt","r")
    data = f.readlines()
    # data = [(int(i[12:i.index(",")]),int(i[i.index(",")+2:i.index(",",i.index(",")+1)])) for i in data[::2]]
    data = [(int(i[1:i.index(",")]),int(i[i.index(",") + 2:-2])) for i in data]
    # print(data)
    f.close()
    scale_list = {}
    pos_list = np.array([[0]])
    from matplotlib import pyplot as plt
    from matplotlib import cm
    from matplotlib import axes
    X = [0]
    Y = [0]
    for i in data:
        part = (i[0],i[1])
        if(part in scale_list):
            scale_list[part] += 1
            pos_list[pos_x][pos_y] += 1
        else:
            scale_list[part] = 1
            if part[0] in X:
                pos_x = X.index(part[0])
            else:
                pos_x = len(X)
                for j in range(len(X)):
                    if(X[j] < part[0]):
                        pos_x = j
                        break
                X.insert(pos_x,part[0])
                to_insert = np.zeros((1,len(Y)))
                pos_list = np.insert(pos_list,pos_x,values=to_insert,axis=0)
            if part[1] in Y:
                pos_y = Y.index(part[1])
            else:
                pos_y = len(Y)
                for j in range(len(Y)):
                    if(Y[j] > part[1]):
                        pos_y = j
                        break
                Y.insert(pos_y,part[1])
                to_insert = np.zeros((1,len(X)))
                pos_list = np.insert(pos_list,pos_y,values=to_insert,axis=1)
            pos_list[pos_x][pos_y] += 1

    X.pop(-1)
    Y.pop(0)
    pos_list = np.delete(pos_list,-1,axis=0)
    pos_list = np.delete(pos_list,0,axis=1)
    
    bin_num = 25
    pos = 0
    num = 0
    pop_pos = []
    list_num = 0
    for i in range(1,len(X)):
        if(abs(X[i] - X[pos]) <= bin_num):
            pos_list[pos-list_num] += pos_list[i - num]
            pos_list = np.delete(pos_list,i - num,axis = 0)
            pop_pos.append(i-num)
            num += 1
            # print(X[pos])
        else:
            X[pos] = X[pos] - bin_num // 2
            pos = i
            list_num = num
    for i in pop_pos:
        X.pop(i)
    pos = 0
    num = 0
    pop_pos = []
    list_num = 0
    for i in range(1,len(Y)):
        if(abs(Y[i] - Y[pos]) <= bin_num):
            pos_list[:,pos-list_num] += pos_list[:,i - num]
            pos_list = np.delete(pos_list,i - num,axis = 1)
            pop_pos.append(i-num)
            num += 1
        else:
            Y[pos] = Y[pos] + bin_num // 2
            pos = i
            list_num = num
    for i in pop_pos:
        Y.pop(i)
    
    print(X)
    print(Y)
    df = pd.DataFrame(pos_list,index = X ,columns = Y)
    print(df.head())
    
    # sns.clustermap(df,annot=False, fmt='d', linewidths=.5, cmap='YlGnBu')
    # plt.show()
    sns.set()
    ax = sns.heatmap(df,annot=False, fmt='d', linewidths=.5, cmap='YlGnBu')
    plt.title("This is the scale of voc2012 bins=" + str(bin_num))
    plt.show()
    df.to_csv("file/VOC2012/voc2012_" + str(bin_num) + ".csv")
    
def draw_mult():
    df0 = pd.read_csv("file/COCO/coco_or_train_50.csv",index_col=0)
    df1 = pd.read_csv("file/COCO/coco_or_train_25.csv",index_col=0)
    df2 = pd.read_csv("file/COCO/coco_or_test_50.csv",index_col=0)
    df3 = pd.read_csv("file/COCO/coco_or_test_25.csv",index_col=0)
    df4 = pd.read_csv("file/COCO/coco_or_val_50.csv",index_col=0)
    df5 = pd.read_csv("file/COCO/coco_or_val_25.csv",index_col=0)
    f, ((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(figsize = (20, 15),nrows=2,ncols=3)
    sns.set()
    ax1.set_title("coco_or_train_50")
    sns.heatmap(df0,annot=False, ax = ax1,fmt='d', linewidths=.5, cmap='YlGnBu')
    ax2.set_title("coco_or_test_50")
    sns.heatmap(df2,annot=False, ax = ax2,fmt='d', linewidths=.5, cmap='YlGnBu')
    ax3.set_title("coco_or_val_50")
    sns.heatmap(df1,annot=False, ax = ax3,fmt='d', linewidths=.5, cmap='YlGnBu')
    ax4.set_title("coco_or_train_25")
    sns.heatmap(df3,annot=False, ax = ax4,fmt='d', linewidths=.5, cmap='YlGnBu')
    ax5.set_title("coco_or_test_25")
    sns.heatmap(df3,annot=False, ax = ax5,fmt='d', linewidths=.5, cmap='YlGnBu')
    ax6.set_title("coco_or_val_25")
    sns.heatmap(df3,annot=False, ax = ax6,fmt='d', linewidths=.5, cmap='YlGnBu')
    plt.show()
    
    
if __name__ == "__main__":
    # d = draw()
    draw_mult()