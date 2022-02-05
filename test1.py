import os

rootPath = '/home/a286/hjs_dir1/EelPred/datasetVer0/Results/'

eachFolderLst = os.listdir(rootPath)

notRemoveLst = [str(100*i)+'.pth' for i in range(100)] + ['Result.png']

for eachFolder in eachFolderLst:
    fullEachFolder = rootPath+eachFolder+'/'
    eachFileLst = os.listdir(fullEachFolder)
    for eachFile in eachFileLst:
        fullEachFilePath = fullEachFolder + eachFile
        if eachFile not in notRemoveLst:
            os.remove(fullEachFilePath)
            print(f'{fullEachFilePath} removed')
