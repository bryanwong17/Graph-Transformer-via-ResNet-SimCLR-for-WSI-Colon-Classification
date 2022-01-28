import os
import shutil
import glob

def main():
    source1 = '../../dataset/tiles/colon/colon_D'
    source2 = '../../dataset/tiles/colon/colon_N'
    target = '../../dataset/tiles/colon/wsi2'
    source1_files = os.listdir(source1)
    source2_files = os.listdir(source2)

    index = []
    for f in source1_files:
        f = f.strip()
        f = f.split('-')[0]
        index.append(f)

    for f in source2_files:
        f = f.strip()
        f = f.split('-')[0].strip()
        index.append(f)
    
    index = list(set(index))
    # print(index)
    for f in source1_files:
        f = f.strip()
        # print(f)
        for i in index:
            if f.split("-")[0] == i:
                if not os.path.exists(os.path.join(target, i)):
                    os.makedirs(os.path.join(target, i))
                shutil.copy(os.path.join(source1, f), os.path.join(target, i, f))
                break
    
    for f in source2_files:
        f = f.strip()
        for i in index:
            if f.split("-")[0].strip() == i:
                if not os.path.exists(os.path.join(target, i)):
                    os.makedirs(os.path.join(target, i))
                shutil.copy(os.path.join(source2, f), os.path.join(target, i, f))
                break


if __name__ == "__main__":
    main()