import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    save_path = "result_crop_resnet18_pretrained/curve"
    if not os.path.isdir(save_path): os.mkdir(save_path)
    temp = np.array([[18,0],[2,22]])
    cm_df = pd.DataFrame(temp, index=["Normal", "Dysplasia"], columns=["Normal", "Dysplasia"])
    print(cm_df)
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt=".1f")
    plt.title("Graph Transformer for WSI Colon Classification")
    plt.ylabel("Predicted Classes")
    plt.xlabel("Actual Classes")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()


if __name__ == "__main__":
    main()