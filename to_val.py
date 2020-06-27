import os

for i in range(17):
    os.system("new_validation.py --voc_path /media/data1/zzy/Datasets/voc2007/ --model_path ./model_final1/model_final1" + str(i) + ".pt")