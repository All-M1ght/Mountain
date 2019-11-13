import pandas as pd
import random

pd.set_option("display.max_colwidth",100)
data = pd.read_csv(r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/7000_8000.csv')
trainFile = r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/case1train.csv'
testFile = r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/case1test.csv'
randomList = []
# for i in range(0, 150):
#     randLine = random.randint(0, 260)
#     while randLine in randomList:
#         randLine = random.randint(0, 260)
#     randomList.append(randLine)
# with open("/Users/allmight/PycharmProjects/Mountain/src/data/data3/case1train"):

for i in range(0, 111):
    string = str(data.loc[i])
    # print(string)
    splited = string.split("    ")
    # print(splited[1])
    line = splited[1].split("\n")[0]
    # if i in randomList:
    #     with open(trainFile, 'a+') as f:
    #         f.write(line + '\n')
    # else:
    #     with open(testFile, 'a+') as f:
    #         f.write(line + '\n')
    with open(trainFile, 'a+') as f:
        f.write(line + '\n')
    # print(line)
    # fname_mname_tall = line.split(";")
    # tall = int(fname_mname_tall[2])
    # if 0 <= tall < 1000:
    #     print(line)
    #     # file.write(line + "\n")
    #     print(tall)
        # string = fname_mname_tall[0]+";"+fname_mname_tall[2]
        # file.write(line+"\n")