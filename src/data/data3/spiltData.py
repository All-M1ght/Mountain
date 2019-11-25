import pandas as pd

pd.set_option("display.max_colwidth",100)
data = pd.read_csv(r'D:\PycharmProject\Mountain\src\data\data3\imagesV0.1.csv')
with open("D:\PycharmProject\Mountain\src\data\data3\\8000+.csv","a+") as file:
    for i in range(0, 5177):
        # print(i)
        string = str(data.loc[i])
        # print(string)
        splited = string.split("    ")
        # print(splited[1])
        line = splited[1].split("\n")[0]
        # print(line)
        fname_mname_tall = line.split(";")
        tall = int(fname_mname_tall[2])
        if 8000 <= tall < 9000:
            print(line)
            file.write(line + "\n")
            # print(tall)
            # string = fname_mname_tall[0]+";"+fname_mname_tall[2]
            # file.write(line+"\n")