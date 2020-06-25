import matplotlib.pyplot as plt  
import math
def draw_bar(data,x,y,the_bins):
    max_data = max(data)
    min_data = min(data)

    # bins = [32*32,64*64,128*128,256*256,512*512,640*640]
    bins = [i for i in range(min_data,max_data+the_bins,the_bins)]
    print(bins)
    bins_len = len(bins)
    num = 0
    bins_num = [0]*bins_len
    for i in data:
        pos = 0
        for j in range(bins_len-1):
            # print(bins[j],bins[j+1])
            if(i >= bins[j] and i <= bins[j+1]):
                pos = j
                break
        if(pos == 0):
            print(i)
        # print(pos)
        bins_num[pos] += 1
    print(bins_num)
    print(num)
    plt.xlabel(x)
    plt.xticks(rotation=45)
    plt.ylabel(y)
    plt.title("VOC2007")
    bins = [str(i) for i in bins]
    plt.bar(bins,bins_num,width=0.5)
    plt.legend()
    for a,b in zip(bins,bins_num):
        plt.text(a, b+0.05, str(b / len(data) * 100)[:4] + "%", ha='center', va= 'bottom',fontsize=7)
    plt.show()

# filename = "Voc_train.txt"
# filename = "Voc_trainval.txt"
# filename = "Voc_test.txt"
filename = "Voc_val.txt"
f = open(filename,"r")
data = f.readlines()
f.close()
data = [i.strip().split(" ,  || ")[1].split(" ") for i in data]
width = [int(i[0]) for i in data]
height = [int(i[1]) for i in data]
area = [int(i[0]) * int(i[1]) for i in data]
num = 0

draw_bar(width,"Width","Num",10,)
draw_bar(height,"Height","Num",10)
draw_bar(area,"Area","Num",10000)