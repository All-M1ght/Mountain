import pandas as pd

data = pd.read_csv(r'/Users/allmight/PycharmProjects/Mountain/FileMountainAltitudeA.csv')

reName = ""
num = 0
for i in range(0,1101):
    string = str(data.loc[i])
    splited = string.split("    ")
    line = splited[1].split("\n")[0]
    print(line)
    fname_mname_tall = line.split(";")
    name = fname_mname_tall[1]
    if name != reName:
        num += 1
        reName = name
    if num == 246:
        break
    # string = fname_mname_tall[0]+";"+fname_mname_tall[2]
    # file.write(string+"\n")
print(str(num))






