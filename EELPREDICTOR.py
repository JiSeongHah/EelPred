import os
import torch
import torch.nn as nn
from MY_MODELS import simpleDNN
from torch.optim import AdamW, Adam,SGD
from torch.nn import MSELoss,L1Loss,HuberLoss
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from DataLoading import MyEelDataset
from save_funcs import createDirectory,mk_name
import numpy as np
import matplotlib.pyplot as plt
import time
import csv

from sklearn.metrics import f1_score
from MY_MODELS import EelPredCNNModel

class EelPredictor(nn.Module):
    def __init__(self,
                 data_folder_dir_trn,
                 data_folder_dir_val,
                 MaxEpoch,
                 lossFuc,
                 gpuUse,
                 labelDir,
                 data_folder_dir_test,
                 modelPlotSaveDir,
                 iter_to_accumul,
                 MaxStep,
                 MaxStepVal,
                 whichModel,
                 backboneOutFeature,
                 LinNum,
                 bSizeTrn= 8,
                 bSizeVal=1,
                 lr=3e-4,
                 eps=1e-9):


        super(EelPredictor,self).__init__()

        self.data_folder_dir_trn = data_folder_dir_trn
        self.data_folder_dir_val = data_folder_dir_val
        self.data_folder_dir_test = data_folder_dir_test

        self.iter_to_accumul = iter_to_accumul
        self.MaxStep = MaxStep

        self.MaxStepVal = MaxStepVal


        self.labelDir = labelDir
        self.gpuUse = gpuUse
        self.whichModel = whichModel
        self.backboneOutFeature = backboneOutFeature
        self.LinNum = LinNum
        self.lossFuc = lossFuc

        self.lr = lr
        self.eps = eps
        self.bSizeTrn = bSizeTrn
        self.bSizeVal = bSizeVal

        self.modelPlotSaveDir = modelPlotSaveDir

        ###################MODEL SETTING###########################
        print('failed loading model, loaded fresh model')
        self.EelModel = EelPredCNNModel(
            modelKind=self.whichModel,
            backboneOutFeature=self.backboneOutFeature,
            LinNum=self.LinNum
        )

        #self.EelModel = nn.Linear(2,10)

        if self.gpuUse == True:
            USE_CUDA = torch.cuda.is_available()
            print(USE_CUDA)
            self.device = torch.device('cuda' if USE_CUDA else 'cpu')
            print('학습을 진행하는 기기:', self.device)
        else:
            self.device = torch.device('cpu')
            print('학습을 진행하는 기기:', self.device)


        self.loss_lst_trn = []
        self.loss_lst_trn_tmp = []
        self.loss_lst_val = []
        self.loss_lst_val_tmp = []

        self.acc_lst_trn = []
        self.acc_lst_trn_tmp = []

        self.acc_lst_val = []
        self.acc_lst_val_tmp = []

        self.num4epoch = 0
        self.MaxEpoch = MaxEpoch

        #self.optimizer = Adam(self.Datacon1Model.parameters(),
        #                      lr=self.lr,  # 학습률
        #                      eps=self.eps  # 0으로 나누는 것을 방지하기 위한 epsilon 값
        #                      )

        self.optimizer = SGD(self.EelModel.parameters(),
                              lr=self.lr  # 학습률
                                # 0으로 나누는 것을 방지하기 위한 epsilon 값
                              )

        MyTrnDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_trn,tLabelDir=self.labelDir,TEST=False)
        self.trainDataloader = DataLoader(MyTrnDataset,batch_size=self.bSizeTrn,shuffle=True)

        MyValDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_val, tLabelDir=self.labelDir,TEST=False)
        self.valLen = int(len(MyValDataset)/self.bSizeVal)
        if self.valLen < self.MaxStepVal:
            self.MaxStepVal = self.valLen
        self.valDataloader = DataLoader(MyValDataset, batch_size=self.bSizeVal, shuffle=False)

        MyTestDataset = MyEelDataset(data_folder_dir=self.data_folder_dir_test,tLabelDir=self.labelDir,TEST=True)
        self.testLen = len(MyTestDataset)
        self.TestDataloader = DataLoader(MyTestDataset,batch_size=1,shuffle=False)

        self.EelModel.to(device=self.device)


    def forward(self,x):

        out = self.EelModel(x)

        return out


    def calLoss(self,logit,label):

        if self.lossFuc == 'L1':
            loss = L1Loss()
        elif self.lossFuc == 'L2':
            loss = MSELoss()
        elif self.lossFuc == 'Huber':
            loss = HuberLoss()
        else:
            loss = MSELoss()



        return loss(logit,label)

    def trainingStep(self,trainingNum):

        self.EelModel.train()
        countNum = 0

        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            globalTime= time.time()

            for _,bInput, bLabel  in self.trainDataloader:

                #print(bLabel.size())

                bLabel = bLabel.float()
                localTime= time.time()

                bInput = bInput.to(self.device)

                bLogit = self.forward(bInput)

                bLogit = bLogit.cpu()

                ResultLoss = self.calLoss(bLogit,bLabel)


                ResultLoss = ResultLoss/self.iter_to_accumul
                ResultLoss.backward()

                self.loss_lst_trn_tmp.append(10000*float(ResultLoss.item()))

                if (countNum + 1) % self.iter_to_accumul == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                if countNum == self.MaxStep:
                    break
                else:
                    countNum += 1

                localTimeElaps = round(time.time() - localTime,2)
                globalTimeElaps = round(time.time() - globalTime,2)

                print(f'globaly {globalTimeElaps} elapsed and locally {localTimeElaps} elapsed for {countNum} / {self.MaxStep}'
                      f' of epoch : {trainingNum}/{self.MaxEpoch}'
                      f' with loss : {10000*float(ResultLoss.item())}')

        self.loss_lst_trn.append(self.iter_to_accumul * np.mean(self.loss_lst_trn_tmp))
        print(f'training complete with mean loss : {self.iter_to_accumul * np.mean(self.loss_lst_trn_tmp)}')
        self.loss_lst_trn_tmp = []

        torch.set_grad_enabled(False)
        self.EelModel.eval()

    def valdatingStep(self,validatingNum):

        self.EelModel.eval()
        countNum = 0
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(False):
            for _,valBInput, valBLabel in self.valDataloader:

                valBInput = valBInput.to(self.device)

                valBLogit = self.forward(valBInput)
                valBLogit = valBLogit.cpu()

                ResultLoss = self.calLoss(valBLogit,valBLabel)

                ResultLoss = ResultLoss / self.iter_to_accumul
                self.loss_lst_val_tmp.append(10000*float(ResultLoss.item()))

                print(f'{countNum}/ {self.MaxStepVal} th val of epoch : {validatingNum} complete with loss : {10000 * float(ResultLoss.item())}')

                if countNum == self.MaxStepVal:
                    break
                else:
                    countNum += 1

            self.loss_lst_val.append(self.iter_to_accumul*np.mean(self.loss_lst_val_tmp))
            print(f'validation complete with mean loss : {self.iter_to_accumul*np.mean(self.loss_lst_val_tmp)}')
            self.loss_lst_val_tmp = []

        torch.set_grad_enabled(True)
        self.EelModel.train()

    def TestStep(self):

        self.EelModel.eval()
        countNum = 0
        self.optimizer.zero_grad()

        ResultDict = dict()

        with torch.set_grad_enabled(False):
            for ImageName,TestBInput in self.TestDataloader:

                TestBInput = (TestBInput.float()).to(self.device)

                TestBLogit = self.forward(TestBInput)
                TestBLogit = TestBLogit.cpu()

                if ImageName not in ResultDict:
                    ResultDict[str(ImageName)] = [100*TestBLogit.item()]
                if ImageName in ResultDict:
                    ResultDict[str(ImageName)].append(100*TestBLogit.item())
                print(f'{countNum} / {self.testLen} Pred done  data : {[str(ImageName),100*TestBLogit]}')
                countNum +=1

        print('Start saving Result.....')

        header = ['ImageDir','AvgWeight']
        with open(self.modelPlotSaveDir+'sample_submission.csv','w') as f:
            wr = csv.writer(f)
            wr.writerow(header)
            for ImageKey in ResultDict.keys():
                wr.writerow([str(ImageKey),np.mean(ResultDict[ImageKey])])
                print(f'appending {ImageKey} with {ResultDict[ImageKey]} complete')


        torch.set_grad_enabled(True)
        self.EelModel.train()


    def START_TRN_VAL(self,epoch):


        print('training step start....')
        self.trainingStep(trainingNum=epoch)
        print('training step complete!')

        print('Validation start.....')
        self.valdatingStep(validatingNum=epoch)
        print('Validation complete!')

        fig = plt.figure()
        ax1 = fig.add_subplot(1, 4, 1)
        ax1.plot(range(len(self.loss_lst_trn)), self.loss_lst_trn)
        ax1.set_title('train loss')

        ax3 = fig.add_subplot(1, 4, 3)
        ax3.plot(range(len(self.loss_lst_val)), self.loss_lst_val)
        ax3.set_title('val loss')


        plt.savefig(self.modelPlotSaveDir +  'Result.png', dpi=300)
        print('saving plot complete!')
        plt.close()

        print(f'num4epoch is : {epoch} and self.max_epoch : {self.MaxEpoch}')