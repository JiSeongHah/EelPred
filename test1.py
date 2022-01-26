import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from PIL import Image
from AreaCalc import calcArea
import os
import json

rootPath = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
trainFolderPath = rootPath +'train/1IfHWGbtMf/'
testFolderPath = rootPath + 'test/'
labelPath = rootPath+'train.csv'


lst = sorted([file for file in os.listdir(trainFolderPath) if file.endswith('.jpg')])
jsonlst = sorted([file for file in os.listdir(trainFolderPath) if file.endswith('.json')])

threshold1 = 50
threshold2 = 10000

xxx = 0
for i,jsonfile in zip(lst,jsonlst):

    img = Image.open(trainFolderPath+i)
    bwimg = np.asarray(img.convert('L'))
    boolarr1 = bwimg > threshold1
    boolarr2 = bwimg < threshold2
    totalbool = boolarr1 * boolarr2

    bwimg = totalbool * bwimg > 0

    if xxx ==0:
        tmparr = bwimg
    else:
        tmparr = tmparr * bwimg
        print(tmparr)

    xxx +=1

xxx = 0
for i, jsonfile in zip(lst, jsonlst):

    img = Image.open(trainFolderPath + i)
    bwimg = np.asarray(img.convert('L'))
    boolarr1 = bwimg > threshold1
    boolarr2 = bwimg < threshold2
    totalbool = boolarr1 * boolarr2


    bwimg = (totalbool * bwimg > 0)*1
    print(np.mean(bwimg))
    plt.imshow(bwimg)
    plt.show()
    bwimg = (bwimg - tmparr*1) >0


    print(np.mean(bwimg))




    plt.imshow(bwimg)
    plt.show()

    xxx +=1
    if xxx > 20:
        break



    # with open(trainFolderPath+jsonfile) as json_file:
    #     each_json = json.load(json_file)
    #
    #
    #
    # for j in each_json['data']:
    #     lstX = list(map(int, j['x']))
    #     lstY = list(map(int, j['y']))
    #
    #     for XX,YY in zip(lstX,lstY):
    #         bwimg[YY,XX] = 1
