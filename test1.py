import csv
import matplotlib.pyplot as plt
import torch
import numpy as np
import json
from PIL import Image
from AreaCalc import calcArea
import os
import json
import os
import torch
import torch.nn as nn
from MY_MODELS import simpleDNN
from torch.optim import AdamW, Adam,SGD
from torch.nn import MSELoss
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from DataLoading import MyEelDataset
from save_funcs import createDirectory,mk_name
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LinearRegression
import csv
from sklearn.metrics import f1_score
from sklearn.svm import SVR
# rootPath = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
# trainFolderPath = rootPath +'train/1IfHWGbtMf/'
# testFolderPath = rootPath + 'test/'
# labelPath = rootPath+'train.csv'
#
#
# lst = sorted([file for file in os.listdir(trainFolderPath) if file.endswith('.jpg')])
# jsonlst = sorted([file for file in os.listdir(trainFolderPath) if file.endswith('.json')])
#
# threshold1 = 5000
# threshold2 = 10000
#
# xxx = 0
# for i,jsonfile in zip(lst,jsonlst):
#
#     img = Image.open(trainFolderPath+i)
#     bwimg = np.asarray(img.convert('L'))
#     boolarr1 = bwimg > threshold1
#     boolarr2 = bwimg < threshold2
#     totalbool = boolarr1 * boolarr2
#
#     bwimg = totalbool * bwimg > 0
#
#     if xxx ==0:
#         tmparr = bwimg
#     else:
#         tmparr = tmparr * bwimg
#         print(tmparr)
#
#     xxx +=1
#
# xxx = 0
# for i, jsonfile in zip(lst, jsonlst):
#
#     img = Image.open(trainFolderPath + i)
#     bwimg = np.asarray(img.convert('L'))
#     boolarr1 = bwimg > threshold1
#     boolarr2 = bwimg < threshold2
#     totalbool = boolarr1 * boolarr2
#
#     bwimg = (totalbool * bwimg > 0)*1
#
#     with open(trainFolderPath+jsonfile) as json_file:
#         each_json = json.load(json_file)
#
#     for j in each_json['data']:
#         lstX = list(map(int, j['x']))
#         lstY = list(map(int, j['y']))
#
#         for XX,YY in zip(lstX,lstY):
#             bwimg[YY,XX] = 1
#
#
#     # print(np.mean(bwimg))
#     plt.imshow(bwimg)
#     plt.show()
#     # bwimg = (bwimg - tmparr*1) >0
#
#     # with open(trainFolderPath+jsonfile) as json_file:
#     #     each_json = json.load(json_file)
#     #
#     #
#     # for j in each_json['data']:
#     #     lstX = list(map(int, j['x']))
#     #     lstY = list(map(int, j['y']))
#     #
#     #     for XX,YY in zip(lstX,lstY):
#     #         bwimg[YY,XX] = 1
#

rootPath = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
trainFolderPath = rootPath +'train/1IfHWGbtMf/'
testFolderPath = rootPath + 'test/'
labelPath = rootPath+'train.csv'

TOTALFOLDERDIR = [rootPath+ 'train/'+ fil for fil in os.listdir(rootPath+'train/')]
print(TOTALFOLDERDIR)


lst = sorted([file for file in os.listdir(trainFolderPath) if file.endswith('.jpg')])
jsonlst = sorted([file for file in os.listdir(trainFolderPath) if file.endswith('.json')])

threshold1 = 50
threshold2Lst = [100]

aaaa = 0

for otherfolder in TOTALFOLDERDIR:
    print(otherfolder,otherfolder,'------------------------------------------------------------------------')
    lst = sorted([file for file in os.listdir(otherfolder) if file.endswith('.jpg')])
    jsonlst = sorted([file for file in os.listdir(otherfolder) if file.endswith('.json')])

    print(jsonlst)

    aaaa += len(lst)
    print('aaaa is : ',aaaa)
    #
    #
    # for threshold2 in threshold2Lst:
    #
    #     xxx = 0
    #
    #     bwimgLst = []
    #     areaLst = []
    #
    #     for i, jsonfile in zip(lst, jsonlst):
    #
    #         areaTmp = []
    #
    #         img = Image.open(trainFolderPath + i)
    #         bwimg = np.asarray(img.convert('L'))
    #
    #         boolarr1 = bwimg > threshold1
    #         boolarr2 = bwimg < threshold2
    #         totalbool = boolarr1 * boolarr2
    #
    #         bwimg = ((totalbool * bwimg) > 0) * 1
    #
    #         # plt.imshow(bwimg)
    #         # plt.show()
    #
    #         bwImg = np.sum(bwimg)
    #
    #         with open(trainFolderPath + jsonfile) as json_file:
    #             each_json = json.load(json_file)
    #
    #         for j in each_json['data']:
    #             lstX = list(map(int, j['x']))
    #             lstY = list(map(int, j['y']))
    #
    #             area = calcArea(lstX, lstY)
    #
    #             areaTmp.append(area)
    #
    #         bwimgLst.append(bwImg)
    #         areaLst.append(np.mean(areaTmp))
    #
    #         areaTmp = []
    #
    #         # print(f'{i}th done')
    #
    #     Maxbwimg = max(bwimgLst)
    #     Maxarea = max(areaLst)
    #
    #     bwimgArr = (np.array(bwimgLst) / Maxbwimg).reshape(-1, 1)
    #     areaArr = (np.array(areaLst) / Maxarea).reshape(-1, 1)
    #     print(bwimgArr)
    #     print(areaArr)
    #
    #     print(f'{threshold1} done')
    #     model = LinearRegression()
    #     print('111111111111111111111111111111111111111111111111111111111111111111111111111111111111111111')
    #     print(bwimgArr.shape, areaArr.shape)
    #     model.fit(bwimgArr, areaArr)
    #     print('scores is : ', model.score(bwimgArr, areaArr))
    #     # plt.scatter(bwimgLst,areaLst)
    #     # plt.show()
    #     plt.scatter(range(len(bwimgArr)), bwimgArr)
    #     plt.xlabel(str(otherfolder))
    #     plt.show()
    #     # plt.scatter(range(len(areaArr)), areaArr)
    #     # plt.show()

#
#
#
#
#
#
#
#
#
#
#
# # import csv
# # import matplotlib.pyplot as plt
# # import torch
# # import numpy as np
# # import json
# # from PIL import Image
# # from AreaCalc import calcArea
# # import os
# #
# # # rootPath = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
# # # trainFolderPath = rootPath +'train/'
# # # testFolderPath = rootPath + 'test'
# # # folderLst = os.listdir(trainFolderPath)
# # # print(folderLst)
# #
# #
# #
# #
# #
# # class MyEelDataset(torch.utils.data.Dataset):
# #
# #     def __init__(self,data_folder_dir,tLabelDir,TRAIN=True):
# #
# #         self.data_folder_dir = data_folder_dir
# #
# #         self.tLabelDir = tLabelDir
# #
# #         self.data_folder_lst = os.listdir(data_folder_dir)
# #
# #         self.TRAIN = TRAIN
# #
# #
# #
# #         self.labelDict = dict()
# #
# #         with open(self.tLabelDir, 'r') as f:
# #             rdr = csv.reader(f)
# #             for line in rdr:
# #                 try:
# #                     self.labelDict[str(line[0])] = float(line[1])
# #                 except:
# #                     self.labelDict[str(line[0])] = line[1]
# #
# #
# #
# #
# #
# #     def __len__(self):
# #         return len(os.listdir(self.data_folder_dir))
# #
# #     def __getitem__(self, idx):
# #
# #         data_folder_name = self.data_folder_lst[idx]
# #         full_data_dir = self.data_folder_dir+data_folder_name+'/'
# #
# #         json_lst = os.listdir(full_data_dir)
# #         json_lst = [file for file in json_lst if file.endswith(".json")]
# #
# #         area_lst = []
# #
# #         area_avg = 0
# #         area_std = 0
# #
# #         for each_json in json_lst:
# #             with open(full_data_dir+each_json) as json_file:
# #                 each_json_data = json.load(json_file)
# #
# #             for j in each_json_data['data']:
# #                 lstX = list(map(int,j['x']))
# #                 lstY = list(map(int,j['y']))
# #
# #                 area = calcArea(lstX,lstY) /10000
# #
# #                 area_lst.append(area)
# #
# #         area_avg = np.mean(area_lst)
# #         area_std = np.std(area_lst)
# #         area_quant = len(area_lst)/1000
# #
# #         input = torch.tensor([area_avg,area_std,area_quant]).float()
# #
# #         if self.TRAIN == True:
# #
# #             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) /100 )
# #
# #             return data_folder_name, area_lst,input, label
# #
# #         if self.TRAIN != True:
# #
# #             return data_folder_name, input
# #
# # # class MyEelDataset(torch.utils.data.Dataset):
# # #
# # #     def __init__(self,data_folder_dir,tLabelDir,TRAIN=True):
# # #
# # #         self.data_folder_dir = data_folder_dir
# # #
# # #         self.tLabelDir = tLabelDir
# # #
# # #         self.data_folder_lst = os.listdir(data_folder_dir)
# # #
# # #         self.TRAIN = TRAIN
# # #
# # #
# # #
# # #         self.labelDict = dict()
# # #
# # #         with open(self.tLabelDir, 'r') as f:
# # #             rdr = csv.reader(f)
# # #             for line in rdr:
# # #                 try:
# # #                     self.labelDict[str(line[0])] = float(line[1])
# # #                 except:
# # #                     self.labelDict[str(line[0])] = line[1]
# # #
# # #
# # #
# # #
# # #
# # #     def __len__(self):
# # #         return len(os.listdir(self.data_folder_dir))
# # #
# # #     def __getitem__(self, idx):
# # #
# # #         data_folder_name = self.data_folder_lst[idx]
# # #         full_data_dir = self.data_folder_dir+data_folder_name+'/'
# # #
# # #         json_lst = os.listdir(full_data_dir)
# # #         json_lst = sorted([file for file in json_lst if file.endswith(".json")])
# # #         #print(json_lst)
# # #
# # #         area_lst = []
# # #
# # #         area_avg = 0
# # #         area_std = 0
# # #
# # #         instanceNum = -1
# # #
# # #         areaTmp = []
# # #
# # #         for each_json in json_lst:
# # #             with open(full_data_dir+each_json) as json_file:
# # #                 each_json_data = json.load(json_file)
# # #
# # #
# # #             if len(each_json_data['data']) != instanceNum and instanceNum == -1:
# # #                 #print('startstartstartstartstartstartstartstartstartstartstartstartstart')
# # #                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
# # #
# # #                 instanceNum = len(each_json_data['data'])
# # #                 #print(f'istanceNum of after change is : {instanceNum}')
# # #
# # #
# # #                 for j in each_json_data['data']:
# # #                     lstX = list(map(int, j['x']))
# # #                     lstY = list(map(int, j['y']))
# # #
# # #                     area = calcArea(lstX, lstY) / 10000
# # #
# # #                     areaTmp.append(area)
# # #
# # #                 #print(f'done 1 with NUm : {instanceNum}')
# # #
# # #             if len(each_json_data['data']) == instanceNum and instanceNum != -1:
# # #                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
# # #                 instanceNum = len(each_json_data['data'])
# # #                 #print(f'istanceNum of after change is : {instanceNum}')
# # #
# # #
# # #                 for j in each_json_data['data']:
# # #                     lstX = list(map(int, j['x']))
# # #                     lstY = list(map(int, j['y']))
# # #
# # #                     area = calcArea(lstX, lstY) / 10000
# # #
# # #                     areaTmp.append(area)
# # #
# # #                 #print(f'done 2 with NUm : {instanceNum}')
# # #
# # #             if len(each_json_data['data']) != instanceNum and instanceNum != -1:
# # #                 #print('--------------------------------------------------------------------------------')
# # #                 #print(f'istanceNum of now is : {instanceNum} but len is : ', len(each_json_data['data']))
# # #
# # #                 area_lst.append(np.mean(areaTmp))
# # #                 #print(areaTmp)
# # #                 areaTmp = []
# # #
# # #                 instanceNum = len(each_json_data['data'])
# # #                 #print(f'istanceNum of after change is : {instanceNum}')
# # #
# # #
# # #                 for j in each_json_data['data']:
# # #                     lstX = list(map(int, j['x']))
# # #                     lstY = list(map(int, j['y']))
# # #
# # #                     area = calcArea(lstX, lstY) / 10000
# # #
# # #                     areaTmp.append(area)
# # #
# # #                 #print(f'done 3 with NUm : {instanceNum}')
# # #
# # #         #print(f'areaTmp is : {areaTmp}')
# # #         area_lst.append(np.mean(areaTmp))
# # #
# # #         # area_lst = sorted(area_lst)
# # #         #
# # #         # plt.plot(range(len(area_lst)),area_lst)
# # #         # plt.show()
# # #
# # #         area_avg = np.mean(area_lst)
# # #         area_std = np.std(area_lst)
# # #         area_quant = len(area_lst)/1000
# # #
# # #         if np.isnan(area_avg) == True or np.isnan(area_std) ==True:
# # #             print(area_avg,area_std,data_folder_name,each_json,each_json_data)
# # #
# # #         input = torch.tensor([area_avg,area_std,area_quant]).float()
# # #
# # #         if self.TRAIN == True:
# # #
# # #             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) /100 )
# # #
# # #             return data_folder_name, input, label
# # #
# # #         if self.TRAIN != True:
# # #
# # #             return data_folder_name, input
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #
# # # class MyEelDataset(torch.utils.data.Dataset):
# # #
# # #     def __init__(self,data_folder_dir,tLabelDir,rangeNum,TRAIN=True):
# # #
# # #         self.data_folder_dir = data_folder_dir
# # #
# # #         self.tLabelDir = tLabelDir
# # #
# # #         self.data_folder_lst = os.listdir(data_folder_dir)
# # #
# # #         self.TRAIN = TRAIN
# # #
# # #         self.rangeNum = rangeNum
# # #
# # #         self.labelDict = dict()
# # #
# # #         with open(self.tLabelDir, 'r') as f:
# # #             rdr = csv.reader(f)
# # #             for line in rdr:
# # #                 try:
# # #                     self.labelDict[str(line[0])] = float(line[1])
# # #                 except:
# # #                     self.labelDict[str(line[0])] = line[1]
# # #
# # #
# # #     def __len__(self):
# # #         return len(os.listdir(self.data_folder_dir))
# # #
# # #     def __getitem__(self, idx):
# # #
# # #         data_folder_name = self.data_folder_lst[idx]
# # #         full_data_dir = self.data_folder_dir+data_folder_name+'/'
# # #
# # #         json_lst = os.listdir(full_data_dir)
# # #         json_lst = sorted([file for file in json_lst if file.endswith(".json")])
# # #         #print(json_lst)
# # #
# # #         area_lst = []
# # #
# # #         area_avg = 0
# # #         area_std = 0
# # #
# # #         instanceNum = -1
# # #
# # #         areaTmp = []
# # #
# # #         for each_json in json_lst:
# # #             with open(full_data_dir+each_json) as json_file:
# # #                 each_json_data = json.load(json_file)
# # #
# # #
# # #             if len(each_json_data['data']) != instanceNum and instanceNum == -1:
# # #                 #print('startstartstartstartstartstartstartstartstartstartstartstartstart')
# # #                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
# # #
# # #                 instanceNum = len(each_json_data['data'])
# # #                 #print(f'istanceNum of after change is : {instanceNum}')
# # #
# # #
# # #                 for j in each_json_data['data']:
# # #                     lstX = list(map(int, j['x']))
# # #                     lstY = list(map(int, j['y']))
# # #
# # #                     area = calcArea(lstX, lstY) / 10000
# # #
# # #                     areaTmp.append(area)
# # #
# # #                 #print(f'done 1 with NUm : {instanceNum}')
# # #
# # #             if len(each_json_data['data']) == instanceNum and instanceNum != -1:
# # #                 #print(f'istanceNum of now is : {instanceNum} but len is : ',len(each_json_data['data']))
# # #                 instanceNum = len(each_json_data['data'])
# # #                 #print(f'istanceNum of after change is : {instanceNum}')
# # #
# # #
# # #                 for j in each_json_data['data']:
# # #                     lstX = list(map(int, j['x']))
# # #                     lstY = list(map(int, j['y']))
# # #
# # #                     area = calcArea(lstX, lstY) / 10000
# # #
# # #                     areaTmp.append(area)
# # #
# # #                 #print(f'done 2 with NUm : {instanceNum}')
# # #
# # #             if len(each_json_data['data']) != instanceNum and instanceNum != -1:
# # #                 #print('--------------------------------------------------------------------------------')
# # #                 #print(f'istanceNum of now is : {instanceNum} but len is : ', len(each_json_data['data']))
# # #
# # #                 area_lst.append(np.mean(areaTmp))
# # #                 #print(areaTmp)
# # #                 areaTmp = []
# # #
# # #                 instanceNum = len(each_json_data['data'])
# # #                 #print(f'istanceNum of after change is : {instanceNum}')
# # #
# # #
# # #                 for j in each_json_data['data']:
# # #                     lstX = list(map(int, j['x']))
# # #                     lstY = list(map(int, j['y']))
# # #
# # #                     area = calcArea(lstX, lstY) / 10000
# # #
# # #                     areaTmp.append(area)
# # #
# # #                 #print(f'done 3 with NUm : {instanceNum}')
# # #
# # #         #print(f'areaTmp is : {areaTmp}')
# # #         area_quant = len(area_lst)
# # #
# # #         #area_lst = area_lst[int(len(area_lst)/2)-self.rangeNum:int(len(area_lst)/2)+self.rangeNum]
# # #
# # #         area_lst.append(np.mean(areaTmp))
# # #
# # #         # area_lst = sorted(area_lst)
# # #         #
# # #         # plt.plot(range(len(area_lst)),area_lst)
# # #         # plt.show()
# # #
# # #         area_avg = np.mean(area_lst)
# # #         area_std = np.std(area_lst)
# # #
# # #
# # #         if np.isnan(area_avg) == True or np.isnan(area_std) ==True:
# # #             print(area_avg,area_std,data_folder_name,each_json,each_json_data)
# # #
# # #         input = torch.tensor([area_avg,area_std]).float()
# # #
# # #         if self.TRAIN == True:
# # #
# # #             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) /100  )
# # #
# # #             return data_folder_name, input, label
# # #
# # #         if self.TRAIN != True:
# # #
# # #             return data_folder_name, input
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# #
# # #
# # # class MyEelDataset(torch.utils.data.Dataset):
# # #     def __init__(self,data_folder_dir,tLabelDir,rangeNum,TRAIN=True):
# # #
# # #         self.data_folder_dir = data_folder_dir
# # #
# # #         self.tLabelDir = tLabelDir
# # #
# # #         self.data_folder_lst = os.listdir(data_folder_dir)
# # #
# # #         self.TRAIN = TRAIN
# # #
# # #         self.avgfilterNum = 0
# # #
# # #         self.rangeNum = rangeNum
# # #
# # #         self.labelDict = dict()
# # #
# # #         with open(self.tLabelDir, 'r') as f:
# # #             rdr = csv.reader(f)
# # #             for line in rdr:
# # #                 try:
# # #                     self.labelDict[str(line[0])] = float(line[1])
# # #                 except:
# # #                     self.labelDict[str(line[0])] = line[1]
# # #
# # #
# # #     def __len__(self):
# # #         return len(os.listdir(self.data_folder_dir))
# # #
# # #     def __getitem__(self, idx):
# # #
# # #         data_folder_name = self.data_folder_lst[idx]
# # #         full_data_dir = self.data_folder_dir+data_folder_name+'/'
# # #
# # #         json_lst = os.listdir(full_data_dir)
# # #         jpg_lst = os.listdir(full_data_dir)
# # #
# # #         json_lst = sorted([file for file in json_lst if file.endswith(".json")])
# # #         jpg_lst = sorted([file for file in jpg_lst if file.endswith(".jpg")])
# # #
# # #
# # #         area_lst = []
# # #
# # #         area_avg = 0
# # #         area_std = 0
# # #
# # #         instanceNum = 0
# # #
# # #         areaTmp = []
# # #
# # #         for each_jpg in jpg_lst:
# # #
# # #             loadedImg = Image.open(full_data_dir+each_jpg)
# # #             Img2blwh = np.asarray(loadedImg.convert('L'))
# # #
# # #             maskedImg =  Img2blwh > self.rangeNum
# # #
# # #
# # #             if instanceNum == 0:
# # #                 tmpArr = maskedImg
# # #
# # #             else:
# # #                 tmpArr = maskedImg * tmpArr
# # #
# # #                 instanceNum +=1
# # #
# # #         for each_jpg,each_json in zip(jpg_lst,json_lst):
# # #
# # #             with open(full_data_dir+each_json) as json_file:
# # #                 loadedJson = json.load(json_file)
# # #
# # #             for j in loadedJson['data']:
# # #                 lstX = list(map(int, j['x']))
# # #                 lstY = list(map(int, j['y']))
# # #
# # #                 partArea = calcArea(lstX, lstY) / 10000
# # #
# # #                 areaTmp.append(partArea)
# # #
# # #             givenArea = np.mean(areaTmp)
# # #
# # #             loadedImg = Image.open(full_data_dir+each_jpg)
# # #             Img2blwh = np.asarray(loadedImg.convert('L'))
# # #             maskedImg =  Img2blwh > self.rangeNum
# # #
# # #             filtedImg = (maskedImg - tmpArr*1) >0
# # #
# # #             meanArea = np.mean(filtedImg) * givenArea
# # #
# # #             area_lst.append(meanArea)
# # #
# # #
# # #
# # #
# # #
# # #
# # #         #print(f'areaTmp is : {areaTmp}')
# # #         area_quant = len(area_lst)
# # #
# # #         #area_lst = area_lst[int(len(area_lst)/2)-self.rangeNum:int(len(area_lst)/2)+self.rangeNum]
# # #
# # #
# # #         # area_lst = sorted(area_lst)
# # #         #
# # #         # plt.plot(range(len(area_lst)),area_lst)
# # #         # plt.show()
# # #
# # #         area_avg = np.mean(area_lst)
# # #         area_std = np.std(area_lst)
# # #
# # #         input = torch.tensor([area_avg,area_std]).float()
# # #
# # #         if self.TRAIN == True:
# # #
# # #             label = torch.tensor(float(self.labelDict[str(data_folder_name)]) )
# # #
# # #             return data_folder_name, input, label
# # #
# # #         if self.TRAIN != True:
# # #
# # #             return data_folder_name, input
# #
# #
# # rootPath = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
# # trainFolderPath = rootPath +'train/'
# # testFolderPath = rootPath + 'test/'
# # labelPath = rootPath+'train.csv'
# #
# #
# #
# # # #
# # # #
# #
# #
# # dt = MyEelDataset(data_folder_dir=trainFolderPath,tLabelDir=labelPath,TRAIN=True)
# # xLst = []
# # yLst = []
# #
# # n = 50
# # N = 500
# # for each_area_lst in dt:
# #     real_area_lst = each_area_lst[1]
# #     each_area_arr = np.array(real_area_lst[1])
# #     MeanLst = []
# #     for i in range(N):
# #         pyobon = np.random.choice(real_area_lst,n)
# #         pyobonMean = np.mean(pyobon)
# #         MeanLst.append(pyobonMean)
# #
# #     print(np.mean(MeanLst),each_area_lst[1][0],float(np.mean(MeanLst))-float(each_area_lst[1][0]))
# #
# #
# #
# #
# #
