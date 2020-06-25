import matplotlib.pyplot as plt  

lh180 = '''
+-------------+-------+-------+-------+-------+
|  categories |   AP  |  APs  |  APm  |  APl  |
+-------------+-------+-------+-------+-------+
|     bus     | 0.342 | 0.133 | 0.592 |  0.31 |
|     dog     | 0.302 |  0.5  | 0.665 | 0.253 |
|    sheep    | 0.546 | 0.519 | 0.773 | 0.545 |
|  aeroplane  | 0.453 | 0.548 | 0.827 | 0.282 |
|   bicycle   | 0.575 | 0.077 | 0.776 | 0.529 |
|     bird    | 0.503 | 0.463 | 0.619 | 0.502 |
|     boat    | 0.492 | 0.533 |  0.65 | 0.368 |
|    bottle   | 0.605 |  0.43 | 0.779 | 0.472 |
|     car     | 0.622 | 0.696 | 0.847 | 0.418 |
|     cat     | 0.256 |  0.5  | 0.879 |  0.22 |
|    chair    |  0.58 | 0.387 | 0.642 | 0.574 |
|     cow     | 0.559 | 0.456 |  0.8  | 0.451 |
| diningtable | 0.093 |  0.0  | 0.176 | 0.094 |
|    horse    | 0.394 | 0.444 | 0.627 | 0.367 |
|  motorbike  | 0.422 |  0.22 | 0.831 | 0.311 |
|    person   | 0.585 | 0.562 | 0.831 | 0.442 |
| pottedplant | 0.499 | 0.313 | 0.695 | 0.413 |
|     sofa    | 0.252 |  0.0  | 0.581 | 0.235 |
|    train    | 0.229 | 0.333 | 0.597 | 0.164 |
|  tvmonitor  | 0.698 | 0.521 |  0.89 | 0.594 |
|     ALL     | 0.508 | 0.513 |  0.75 | 0.379 |
+-------------+-------+-------+-------+-------+
'''
lh = '''
+-------------+-------+-------+-------+-------+
|  categories |   AP  |  APs  |  APm  |  APl  |
+-------------+-------+-------+-------+-------+
|     bus     | 0.524 | 0.333 | 0.525 |  0.663 |
|     dog     | 0.628 |  0.5  | 0.479 | 0.684 |
|    sheep    | 0.277 | 0.293 | 0.505 | 0.357 |
|  aeroplane  | 0.618 | 0.402 | 0.785 | 0.694 |
|   bicycle   | 0.597 | 0.096 | 0.729 | 0.761 |
|     bird    | 0.547 | 0.204 | 0.554 | 0.7 |
|     boat    | 0.403 | 0.387 |  0.418 | 0.518 |
|    bottle   | 0.446 |  0.27 | 0.647 | 0.714 |
|     car     | 0.633 | 0.54 | 0.67 | 0.692 |
|     cat     | 0.713 |  0.75  | 0.742 |  0.763 |
|    chair    | 0.355 | 0.236 | 0.43 | 0.38 |
|     cow     | 0.438 | 0.575 |  0.696  | 0.74 |
| diningtable | 0.312 |  0.0  | 0.146 | 0.413 |
|    horse    | 0.675 | 0.622 | 0.616 | 0.812 |
|  motorbike  | 0.526 |  0.138 | 0.388 | 0.644|
|    person   | 0.645 |  0.48 | 0.625 | 0.722 |
| pottedplant |  0.35 | 0.387 | 0.443 | 0.454 |
|     sofa    | 0.393 |  0.0  | 0.28 | 0.459 |
|    train    | 0.645 |  0.0  | 0.61 | 0.745 |
|  tvmonitor  | 0.658 | 0.061 |  0.774 | 0.718 |
|     ALL     | 0.542 | 0.403 |  0.579 | 0.641 |
+-------------+-------+-------+-------+-------+
'''
hh = '''
+-------------+-------+-------+-------+-------+
|  categories |   AP  |  APs  |  APm  |  APl  |
+-------------+-------+-------+-------+-------+
|     bus     | 0.761 | 0.29 | 0.736 |  0.841 |
|     dog     | 0.867 |  0.5  | 0.736 | 0.913 |
|    sheep    | 0.745 | 0.705 | 0.8 | 0.828 |
|  aeroplane  | 0.825 | 0.586 | 0.855 | 0.86 |
|   bicycle   | 0.826 | 0.069 | 0.821 | 0.81 |
|     bird    | 0.781 | 0.536 | 0.661 | 0.943 |
|     boat    | 0.695 | 0.645 |  0.722 | 0.746 |
|    bottle   | 0.736 |  0.546 | 0.82 | 0.891 |
|     car     | 0.857 | 0.75 | 0.862 | 0.917 |
|     cat     | 0.902 |  0.5  | 0.806 |  0.926 |
|    chair    | 0.668 | 0.54 | 0.692 | 0.7 |
|     cow     | 0.752 | 0.63 |  0.819  | 0.825 |
| diningtable | 0.64 |  0.0  | 0.291 | 0.754 |
|    horse    | 0.804 | 0.444 | 0.642 | 0.851 |
|  motorbike  | 0.84 |  0.475 | 0.872 | 0.87|
|    person   | 0.828 |  0.602 | 0.82 | 0.893 |
| pottedplant |  0.658 | 0.376 | 0.715 | 0.739 |
|     sofa    | 0.71 |  0.0 | 0.615 | 0.765 |
|    train    | 0.784 |  0.667  | 0.582 | 0.831 |
|  tvmonitor  | 0.852 | 0.616 |  0.863 | 0.901 |
|     ALL     | 0.789 | 0.598 |  0.779 | 0.861 |
+-------------+-------+-------+-------+-------+
'''

def deal_table(table):
    table = table.split("\n")
    name = []
    L = [[],[],[]]
    for i in range(4,len(table)-2):
        table[i] = table[i].split("|")
        name.append(table[i][1].strip(" "))
        print(table[i])
        L[0].append(float(table[i][3]))
        L[1].append(float(table[i][4]))
        L[2].append(float(table[i][5]))
    return L,name
    
def draw_first(L,the_name):
    opacity = 0.6
    color = ["b","orange","g"]
    width = 0.2
    for num,i in enumerate(L):
        pos = [i + num * width for i in range(len(name))]
        plt.bar(pos,i,width = width,alpha = opacity,tick_label=name,fc=color[num],label=["Small","Mid","Large"][num])
    plt.xlabel("categories")
    plt.ylabel(the_name + " AP")
    plt.legend()
    plt.show()
    
def draw_second(L,the_name):
    opacity = 0.6
    color = ["b","orange","g"]
    width = 0.2
    for num,i in enumerate(L):
        pos = [i + num * width for i in range(len(name))]
        plt.bar(pos,i,width = width,alpha = opacity,tick_label=name,fc=color[num],label=["lh","hh","hh180"][num])
    plt.xlabel("categories")
    plt.ylabel(the_name + " AP")
    plt.legend()
    plt.show()
Lhh180, name = deal_table(lh180)
Llh   , name = deal_table(lh)
Lhh   , name = deal_table(hh)
# draw_first(Llh,"lh")
# draw_first(Lhh,"hh")
# draw_first(Lhh180,"hh180")

draw_second([Llh[0],Lhh[0],Lhh180[0]],"Small")
draw_second([Llh[1],Lhh[1],Lhh180[1]],"Middle")
draw_second([Llh[2],Lhh[2],Lhh180[2]],"Large")