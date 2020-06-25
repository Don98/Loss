import os

def deal_cla(position,i,data):
    tmp = "".join([j.strip().replace(", ",",") for j in data[position[i]:position[i+1]]])
    tmp = tmp.split(" ")
    # print(tmp)
    start = tmp[0].index("[")
    end = tmp[0].index("]")
    tmp[0] = tmp[0][start:end+1]
    start = tmp[2].index("[")
    end = tmp[2].index("]")
    tmp[2] = tmp[2][start:end+1]
    tmp = " ".join(tmp).replace(",",", ")
    return tmp + "\n"

def deal_reg(position,i,data):
    tmp = "".join([j.strip().replace(", ",",") for j in data[position[i]:position[i+1]]])
    tmp = tmp.split(" ")
    start = tmp[1].index("[")
    end = tmp[1].index("]")
    tmp[1] = tmp[1][start:end+1]
    return " ".join(tmp) + "\n"

def deal_class(data):
    pos = 0;pos1 = 0
    position  = []
    position1 = []
    for i in range(len(data)):
        if data[i].startswith("tensor"):
        # if pos != 0 and i - pos >= 2 and data[i].startswith("["):
            position.append(i)
        if data[i].strip() == "="*50:
            pos = i
        if data[i].strip() == "-"*50:
            pos1 = i
        if pos1 != 0 and i - pos1 >= 2:
            if not (data[i].startswith(" ")):
                position1.append(i)
    # position1.append(len(data) - 2)
    position.append(pos1)
    new_data = data[:pos+2]
    for i in range(len(position) - 1):
        new_data.append(deal_cla(position,i,data))
    new_data.append(data[pos1])
    new_data.append(data[pos1+1])
    for i in range(len(position1) - 1):
        new_data.append(deal_reg(position1,i,data))
        
    # print(new_data)
    new_data.append(data[-1])
    return new_data

def to_do(name):
    filelist = len(os.listdir(name))
    for i in range(filelist-2501):
        os.remove(name + "record" + str(i) + ".txt")
    filelist = len(os.listdir(name))
    # print(filelist)
    num = 0
    for i in range(84,filelist + 84):
        os.rename(name + "record" + str(i) + ".txt",name + "record" + str(num) + ".txt")
        num += 1
    
def to_do1(name):
    for i in range(2501):
        f = open(name + "record" + str(i) + ".txt","r")
        data = f.readlines()
        f.close()
        pos = 0
        for j in range(len(data)):
            if data[j].strip() == "="*50:
                pos = j
                break
        tmp = data[:pos]
        new_tmp = []
        for j in range(len(tmp)):
            tmp[j] = tmp[j].strip()
            if len(tmp[j]) >= 5:
                if tmp[j][-1] == ",":
                    tmp[j] = tmp[j][:-1]
                new_tmp.append(tmp[j] + "\n")
        data[:pos] = new_tmp
        f = open(name + "record" + str(i) + ".txt","w")
        for i in data:
            f.write(i)
        f.close()
        # exit()
    
    
if __name__ == "__main__":
    for k in range(17):
        name = "Epoch" + str(k) + "/"
        to_do1(name)
    # name = "Epoch3/"
    # to_do(name)
    exit()
    for i in range(2501):
        f = open(name + "/record" + str(i) + ".txt","r")
        data = f.readlines()
        f.close()
        # print(data)
        data = deal_class(data)
        # print(data)
        f = open(name + "/record" + str(i) + ".txt","w")
        for j in data:
            f.write(j)
        f.close()
        # exit()