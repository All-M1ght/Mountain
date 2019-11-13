import pandas as pd
import random

pd.set_option("display.max_colwidth",100)
data = pd.read_csv(r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/8000+.csv')
trainFile = r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/case2train.csv'
testFile = r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/case2test.csv'

num = 0
reName = ""
start = 0
table = {}
pre = 0
countLine = 111 #init
count = 110 #init
imgcount = 0
randomList = []
trainlist = []
for i in range(0, countLine):
    string = str(data.loc[i])
    splited = string.split("    ")
    line = splited[1].split("\n")[0]
    fname_mname_tall = line.split(";")
    name = fname_mname_tall[1]
    if name != reName:
        num += 1
        if i > 0 :
            len = int(i)-int(pre)
            table[num-1] = str(pre)+"_"+str(i-1)+"_"+str(len)
        reName = name
        pre = i
table[num] = str(pre) + "_" + str(countLine - 1) + "_" + str(countLine-int(pre))


print(num)

while (imgcount - count) < -5 or (imgcount - count) > 5:
    randLine = random.randint(1, num)
    while randLine in randomList:
        randLine = random.randint(1, num)
    randomList.append(randLine)
    # print(randLine)
    # print(table[randLine])
    imgcount += int(table[randLine].split("_")[2])

print(imgcount)

for i in randomList:
    start_end_len = table[i]
    start = start_end_len.split("_")[0]
    end = start_end_len.split("_")[1]
    len = start_end_len.split("_")[2]
    for j in range(int(start),int(end)+1):
        trainlist.append(j)

print(trainlist.__len__())

for i in range(0, countLine):
    string = str(data.loc[i])
    # print(string)
    splited = string.split("    ")
    # print(splited[1])
    line = splited[1].split("\n")[0]
    if i in trainlist:
        with open(trainFile, 'a+') as f:
            f.write(line + '\n')
    else:
        with open(testFile, 'a+') as f:
            f.write(line + '\n')
    # with open(trainFile, 'a+') as f:
    #     f.write(line + '\n')


