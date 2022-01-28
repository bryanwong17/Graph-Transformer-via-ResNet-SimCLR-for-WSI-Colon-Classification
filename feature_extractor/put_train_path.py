import os
from os import listdir
from os.path import isfile, join
import csv
import pandas as pd

# D:0
# M:1
# N:2

def main():

    columns = ["path to patches"]
    PATH_FILE = "../../dataset/tiles/colon/train_wsi"
    
    # files_N = [f for f in listdir(PATH_FILE + "/colon_N") if isfile(join(PATH_FILE + "/colon_N", f))]
    # files_D = [f for f in listdir(PATH_FILE + "/colon_D") if isfile(join(PATH_FILE + "/colon_D", f))]
    # files_M = [f for f in listdir(PATH_FILE + "/2m") if isfile(join(PATH_FILE + "/2m", f))]

    with open("train_path_patches.csv", 'w', newline="") as f:
        writer = csv.writer(f)

        # write the name of column
        writer.writerow(columns)

        wsi_files = os.listdir(PATH_FILE)
        for f in wsi_files:
            temp = []
            each_wsi = os.listdir(os.path.join(PATH_FILE, f))
            for patch in each_wsi:
                temp = []
                temp.append(patch)
                writer.writerow(temp)

        # write the data for N
        # for i in range(len(files_N)):
        #     temp = []
        #     temp.append(files_N[i])
        #     # temp.append(0)
        #     writer.writerow(temp)

        # write the data for D
        # for i in range(len(files_D)):
        #     temp = []
        #     temp.append(files_D[i])
        #     # temp.append(1)
        #     writer.writerow(temp)
        
        # write the data for M
        # for i in range(len(files_M)):
        #     temp = []
        #     temp.append(files_M[i])
        #     temp.append(2)
        #     writer.writerow(temp)

    # to random data
    # df = pd.read_csv("train_path_patches.csv")
    # df = df.sample(frac=1)
    # df.to_csv("all_path_patches.csv", index=False)


if __name__ == "__main__":
    main()



