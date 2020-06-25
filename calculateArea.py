import numpy as np

if __name__ == "__main__":
    f = open("file/VOC2012/VOC2012.txt", "r")
    data = f.readlines()
    data = [[int(i[1:i.index(",")]), int(i[i.index(",") + 2:-2])] for i in data[::2]]
    print(data)
    area = [0 for i in range(len(data))]
    for i, coor in enumerate(data):
        area[i] = coor[0] * coor[1]
    np.savetxt('file/VOC2012/VOC2012_area.txt', area, fmt='%d')
    print(area)