import os
import shutil

rootPath = '/home/a286winteriscoming/Downloads/EelPred/dataset/dataset/'
trainFolderPath = rootPath +'train/'
testFolderPath = rootPath + 'test/'
labelPath = rootPath+'train.csv'

trnLst = os.listdir(trainFolderPath)
for eachDir in trnLst:
    totalEachDir = trainFolderPath + eachDir+'/'
    eachLst = os.listdir(totalEachDir)
    for eachFile in eachLst:
        totalEachFile = totalEachDir+eachFile
        newFilename = eachDir+'_'+eachFile
        #print(totalEachFile)
        #print(trainFolderPath+newFilename)
        shutil.move(totalEachFile,trainFolderPath+newFilename)
        print(f'moving file from {totalEachFile} to {trainFolderPath+newFilename} complete')

