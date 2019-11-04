import pandas as pd

data = pd.read_csv(r'C:\Users\yifan_hu\Desktop\dataset\\regression\FileMountainAltitudeA.csv')
with open("D:\PycharmProject\Mountain\src\data\\regression\case1test","w") as file:
    for i in range(771,1101):
        string = str(data.loc[i])
        splited = string.split("    ")
        line = splited[1].split("\n")[0]
        # print(line)
        fname_mname_tall = line.split(";")
        string = fname_mname_tall[0]+";"+fname_mname_tall[2]
        file.write(string+"\n")
# string = str(data.loc[1])
# print(string)
# splited = string.split("    ")
# print(len(splited))
# print(splited[1])
# # line = splited[2].split("\n")[0]
# # print(line)
# # fname_mname_tall = line.split(";")
# # string = fname_mname_tall[0]+";"+fname_mname_tall[2]





