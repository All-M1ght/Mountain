import pandas as pd
pd.set_option("display.max_colwidth",100)
data = pd.read_csv(r'/Users/allmight/PycharmProjects/Mountain/src/data/data3/case2train.csv')
with open("/Users/allmight/PycharmProjects/Mountain/src/data/data3/case2train.txt", "w") as file:
    for i in range(0, 1795):
        string = str(data.loc[i])
        splited = string.split("    ")
        line = splited[1].split("\n")[0]
        print(line)
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





