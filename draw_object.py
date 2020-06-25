import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from To_cluster import cluster_model
from math import *

def draw_object_map(data = None):
    if data == None:
        f = open("file/COCO/train_instance.txt","r")
        data = f.readlines()
        data = [float(i[1:-2].split(", ")[-2]) * float(i[1:-2].split(", ")[-1]) for i in data]
    # print(data)
        f.close()
    max_data = max(data)
    min_data = min(data)

    bins = [32*32,64*64,128*128,256*256,512*512,640*640]
    # bins = [2*2,4*4,6*6,8*8,12*12,16*16,20*20,24*24,28*28,32*32,48*48,64*64,80*80,96*96,128*128,224*224,256*256,512*512,640*640]
    the_range = []
    bins_len = len(bins)
    for i in range(bins_len-1):
        the_range.append(bins[i-1] / (bins[i] + bins[i-1]))
    bins_num = [0]*bins_len
    for i in data:
        pos = 0
        for j in range(bins_len-1):
            if(i >= bins[j] and i <= bins[j+1]):
                pos = j
                break
        if(i - bins[j] <= the_range[pos]):
            bins_num[pos] += 1
        else:
            bins_num[pos+1] += 1
    bins = [str(sqrt(i))[:-2] + "*" + str(sqrt(i))[:-2] for i in bins]
    print(bins)
    print(bins_num)
    plt.bar(bins,bins_num)
    for a,b in zip(bins,bins_num):
        plt.text(a, b+0.05, str(b / len(data) * 100)[:4] + "%", ha='center', va= 'bottom',fontsize=7)
    plt.show()

# def two_class():
    # f = open("file/COCO/train_instance.txt","r")
    # data = f.readlines()
    # data = [float(i[1:-2].split(", ")[-2]) * float(i[1:-2].split(", ")[-1])

if __name__ == "__main__":
    draw_object_map()
    two_class()