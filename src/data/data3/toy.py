import pandas as pd
import random

pd.set_option("display.max_colwidth",100)
data = pd.read_csv(r'D:\PycharmProject\Mountain\src\data\data3\8000+.csv')
trainFile = r'D:\PycharmProject\Mountain\src\data\data3\case1train.csv'
testFile = r'D:\PycharmProject\Mountain\src\data\data3\case1test.csv'
randomList = []
rangeRandom = 50
countLine = 104
for i in range(0, rangeRandom):
    randLine = random.randint(0, countLine-1)
    while randLine in randomList:
        randLine = random.randint(0, countLine-1)
    randomList.append(randLine)
# with open("/Users/allmight/PycharmProjects/Mountain/src/data/data3/case1train"):

for i in range(0, countLine):
    string = str(data.loc[i])
    # print(string)
    splited = string.split("    ")
    # print(splited[1])
    line = splited[1].split("\n")[0]
    if i in randomList:
        with open(trainFile, 'a+') as f:
            f.write(line + '\n')
    else:
        with open(testFile, 'a+') as f:
            f.write(line + '\n')
    # with open(trainFile, 'a+') as f:
    #     f.write(line + '\n')
    # print(line)
    # fname_mname_tall = line.split(";")
    # tall = int(fname_mname_tall[2])
    # if 0 <= tall < 1000:
    #     print(line)
    #     # file.write(line + "\n")
    #     print(tall)
        # string = fname_mname_tall[0]+";"+fname_mname_tall[2]
        # file.write(line+"\n")