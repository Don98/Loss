import pandas as pd
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt

def cluster_model(x,model,num):
    if(model == "dbscan"):
        dbscan = DBSCAN(eps = num, min_samples = 5,metric = 'euclidean').fit(x)
        dbscan.fit(x)
        return dbscan.labels_,dbscan.cluster_centers_
    if (model == "kmeans"):
        kms = KMeans(n_clusters=num)
        return kms.fit_predict(x),kms.cluster_centers_ 


if __name__ == "__main__":
    f = open("file/COCO/coco_result.txt", "r")
    # f = open("file/VOC2007/voc_result.txt", "r")
    data = f.readlines()
    data = [i.strip().split(" || ")[1] for i in data]
    data = [(int(i.split(" ")[0]),int(i.split(" ")[1])) for i in data]
    data = [i if i[0] < i[1] else [i[1],i[0]]for i in data]#对折
    f.close()
    x = np.array(data)
    
    y = cluster_model(x, "kmeans",2)
    center = y[1]
    print(center)
    y = y[0]
    # y = cluster_model(x,"dbscan",10)[0]
    y = np.array(y)
    fig = plt.figure()
    ax = plt.subplot()
    ax.scatter(x[y == 0][:, 0], x[y == 0][:, 1], alpha=0.5)
    ax.scatter(x[y == 1][:, 0], x[y == 1][:, 1], c='green', alpha=0.5)
    plt.show()
    with open("coco_center_kmeans.txt","w") as f:
        f.write(str(center[0])+"\n")
        f.write(str(center[1])+"\n")