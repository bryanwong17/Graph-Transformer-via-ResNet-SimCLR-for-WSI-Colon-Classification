import os
import shutil
import glob

def main():
    source = '../graphs/simclr_files/'
    source_files = os.listdir(source)

    for f in source_files:
        f_strip = f.strip()
        os.rename(source + f, source + f_strip)

if __name__ == "__main__":
    main()