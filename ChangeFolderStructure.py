import os
import shutil

rootPath = '/home/a286winteriscoming/Downloads/EelPred/datasetVer1/dataset/'
trainFolderPath = rootPath +'train/'
valFolderPath = rootPath + 'val/'
testFolderPath = rootPath + 'test/'
labelPath = rootPath+'train.csv'

#trnLst = os.listdir(trainFolderPath)
valLst = os.listdir(valFolderPath)

testLst = os.listdir(testFolderPath)
for eachDir in testLst:
    totalEachDir = testFolderPath + eachDir+'/'
    eachLst = os.listdir(totalEachDir)
    for eachFile in eachLst:
        totalEachFile = totalEachDir+eachFile
        newFilename = eachDir+'_'+eachFile
        #print(totalEachFile)
        #print(trainFolderPath+newFilename)
        shutil.move(totalEachFile,testFolderPath+newFilename)
        print(f'moving file from {totalEachFile} to {testFolderPath+newFilename} complete')

