import numpy as np
import math
import matplotlib.pyplot as plt

def get_center(path):
    f = open(path,"r")
    data = f.readlines()
    data = [(float(i[1:i.index(" ")]),float(i[i.index(" ")+1:-2])) for i in data]
    f.close()
    return data

def get_dis(x,y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1]-y[1]) ** 2)

def get_target_partion(data,center,target1,target2):
    distance = 150
    y = [0] * len(data)
    for i in range(len(data)):
        if(get_dis(data[i],center[0]) <= distance):
            y[i] += 1
        if(get_dis(data[i],center[1]) <= distance):
            y[i] += 2
    print("未分类占据比例：",y.count(0) / len(data))
    print("两者皆是分为占全部比例：",y.count(3) / len(data))
    print("两者皆是占第一类比例：",y.count(3) / (y.count(1) + y.count(3)))
    print("两者皆是占第二类比例：",y.count(3) / (y.count(2) + y.count(3)))
    print("第一类占全部比例比例：",(y.count(3) + y.count(1))/ len(data))
    print("第二类占全部比例比例：",(y.count(3) + y.count(2))/ len(data))
    print("第一二类占比比例：",(y.count(3) + y.count(1))/ (y.count(3) + y.count(2)))
    # print(y.count(0))
    # print(y.count(1)+y.count(3))
    # print(y.count(2)+y.count(3))
    # print(y.count(3))
    # print(len(data))
    return y

if __name__ == "__main__":
    f = open("file/COCO/scale_h_w_or.txt","r")
    center_path = "center_kmeans.txt"
    data = f.readlines()
    f.close()
    data = [[int(i[1:i.index(",")]), int(i[i.index(",") + 2:-2])] for i in data]
    data = [i if i[0] < i[1] else [i[1],i[0]]for i in data]#对折
    x = np.array(data)
    center = get_center("center_kmeans.txt")
    y = get_target_partion(data,center,0.05,0.2)
    
    y = np.array(y)
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(x[y == 0][:, 0], x[y == 0][:, 1], c='red', alpha=0.5)
    ax.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c='green', alpha=0.5)
    ax.scatter(x[y == 2][:, 0], x[y == 2][:, 1], c='blue',  alpha=0.5)
    ax.scatter(x[y == 3][:, 0], x[y == 3][:, 1], c='yellow',  alpha=0.5)
    plt.show()