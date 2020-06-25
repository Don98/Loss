import matplotlib.pyplot as plt  
from prettytable import PrettyTable

def read_data(result,name):
    f = open(result,"r")
    data = f.readlines()

    file = open(name, 'r') 
    js = file.read()
    a = js[1:-1].split(", ")
    result = []
    for i in a:
        i = i.split(": ")
        result.append(i[1][1:-1])
    file.close()
    f.close()
    return data,result

def get_data(data,result):
    sma = [0] * len(result)
    mid = [0] * len(result)
    lar = [0] * len(result)
    small_bboxes=[];middle_bboxes=[];large_bboxes=[];all_bboxes=[]
    small_pic=[];middle_pic=[];large_pic=[]
    for i in data:
        i = i.split(" ,  || ")
        width , height = i[1].strip().split(" ")
        objects = i[0].split(" , ")
        before_num = [0,0,0]
        if(len(objects[0]) <= 5):
            continue
        for j in range(len(objects)):
            objects[j] = objects[j].split(" , ")[0].split(" c: ")
            bboxes = objects[j][0].split(" ")
            # print(bboxes)
            bboxes = float(bboxes[0]) * float(bboxes[1])
            objects[j] = [bboxes,objects[j][1]]
            if bboxes <= 32 * 32:
                small_bboxes.append(objects[j])
                before_num[0] += 1
                sma[result.index(objects[j][1])] += 1
            elif bboxes <= 96*96:
                middle_bboxes.append(objects[j])
                before_num[1] += 1
                mid[result.index(objects[j][1])] += 1
            else:
                large_bboxes.append(objects[j])
                before_num[2] += 1
                lar[result.index(objects[j][1])] += 1
            all_bboxes.append(objects[j])
        if before_num[0] > 0:
            small_pic.append(" ")
        if before_num[1] > 0:
            middle_pic.append(" ")
        if before_num[2] > 0:
            large_pic.append(" ")
    return sma,mid,lar,small_bboxes,middle_bboxes,large_bboxes,small_pic,middle_pic,large_pic,all_bboxes

def get_print(sma,mid,lar,small_bboxes,middle_bboxes,large_bboxes,small_pic,middle_pic,large_pic,all_bboxes):
    print(len(small_bboxes),len(middle_bboxes),len(small_bboxes),len(all_bboxes))
    print(len(small_pic),len(middle_pic),len(large_pic),len(data))
    print(len(small_bboxes) / len(all_bboxes)*100,len(middle_bboxes) / len(all_bboxes)*100,len(large_bboxes) / len(all_bboxes)*100)
    print(len(small_pic) / len(data)*100,len(middle_pic) / len(data)*100,len(large_pic) / len(data)*100)

def draw_s_m_l(name,sma,mid,lar):
    # draw small 、middle 、 large
    L = [0,0,0]
    # plt.figure(figsize=(200,200))
    for num,i in enumerate(name):
        now = [sma[num],mid[num],lar[num]]
        # if num <= 71:
            # i = ""
        plt.bar(["small","middle","large"],now,bottom = L,label = i)
        L = [L[i] + now[i] for i in range(3)]
    plt.xlabel("scale")
    plt.ylabel("number")
    plt.legend()
    plt.show()
    
def draw_all_classes(name,sma,mid,lar):
    # draw all classes
            
    L = [0]*len(name)
    plt.bar(name,sma,bottom = L,label = "small",alpha = 0.6)
    L = [L[i] + sma[i] for i in range(len(name))]
    plt.bar(name,mid,bottom = L,label = "middle",alpha = 0.6)
    L = [L[i] + mid[i] for i in range(len(name))]
    plt.bar(name,lar,bottom = L,label = "large",alpha = 0.6)
    plt.xlabel("categories",fontsize=1)
    plt.xticks(rotation=90)
    plt.ylabel("number")
    plt.legend()
    plt.tight_layout()
    plt.show()

def draw_classes_sml(name,sma,mid,lar):    
    opacity = 0.6
    color = ["b","orange","g"]
    width = 0.2
    sma.append(sum(sma)) 
    mid.append(sum(mid)) 
    lar.append(sum(lar)) 
    name.append("ALL")
    for num,i in enumerate([sma,mid,lar]):
        L = [i + num * width for i in range(len(name))]
        plt.bar(L,i,width = width,alpha = opacity,tick_label=name,fc=color[num],label=["Small","Mid","Large"][num])
    plt.xlabel("categories")
    plt.ylabel("number")
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()

def write_to_file(filename,name):
    f = open(filename,"w")
    f.write(" ," + ",".join(name) + "\n")
    print(name)
    f.write("ALL (%),")
    f.write(",".join([str(round((sma[i]+mid[i]+lar[i])/len(all_bboxes)*100,1)) for i in range(len(name))])+"\n")
    print([(sma[i]+mid[i]+lar[i]) for i in range(len(name))])
    f.write("Small (%),")
    f.write(",".join([str(round(i/len(all_bboxes)*100,1)) for i in sma])+"\n")
    print([i for i in sma])
    f.write("Mid (%),")
    f.write(",".join([str(round(i/len(all_bboxes)*100,1)) for i in mid])+"\n")
    print([i for i in mid])
    f.write("Large (%),")
    f.write(",".join([str(round(i/len(all_bboxes)*100,1)) for i in lar])+"\n")
    print([i for i in lar])
    
    f.write("\n\n")
    f.write("ALL: ,")
    f.write(",".join([str(sma[i]+mid[i]+lar[i]) for i in range(len(name))])+"\n")
    f.write("Small ,")
    f.write(",".join([str(i) for i in sma])+"\n")
    f.write("Mid ,")
    f.write(",".join([str(i) for i in mid])+"\n")
    print([i for i in mid])
    f.write("Large ,")
    f.write(",".join([str(i) for i in lar])+"\n")
    
    f.close()

def draw_table(sma,mid,lar,small_bboxes,middle_bboxes,large_bboxes,small_pic,middle_pic,large_pic,all_bboxes,name):
    table = PrettyTable([' ','Small','Mid','Large'])
    table.add_row(["Ratio of total boxes(%)",round(len(small_bboxes) / len(all_bboxes)*100,3),round(len(middle_bboxes) / len(all_bboxes)*100,3),round(len(large_bboxes) / len(all_bboxes)*100,3)])
    table.add_row(["Ratio of images included(%)",round(len(small_pic) / len(data)*100,3),round(len(middle_pic) / len(data)*100,3),round(len(large_pic) / len(data)*100,3)])
    print(table)
    table = PrettyTable([" "] +name)
    table.add_row(["ALL"] + [(sma[i]+mid[i]+lar[i]) for i in range(len(name))])
    table.add_row(["Small"] + sma)
    table.add_row(["Mid"] + mid)
    table.add_row(["Large"] + lar)
    print(table)

if __name__ == "__main__":
    data ,result  = read_data("file/VOC2007/voc_val.txt","file/VOC2007/VOC_classes.txt")
    # data ,result  = read_data("file/COCO/COCO_result.txt","file/COCO/COCO_classes.txt")
    sma,mid,lar,small_bboxes,middle_bboxes,large_bboxes,small_pic,middle_pic,large_pic ,all_bboxes= get_data(data,result)
    # get_print(sma,mid,lar,small_bboxes,middle_bboxes,large_bboxes,small_pic,middle_pic,large_pic,all_bboxes)
    # draw_s_m_l(result,sma,mid,lar)
    # draw_all_classes(result,sma,mid,lar)
    draw_classes_sml(result,sma,mid,lar)
    # write_to_file("file/VOC2007/train/VOC_val.csv",result)
    # write_to_file("COCO_sta.csv",result)
    # draw_table(sma,mid,lar,small_bboxes,middle_bboxes,large_bboxes,small_pic,middle_pic,large_pic,all_bboxes,result)