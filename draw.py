import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix

def get_roc_curve(save_path, label_actual, predicted_score):
    fpr = {}
    tpr = {}
    roc_auc = {}

    n_classes = 2

    label_actual = np.array(label_actual)
    predicted_score = np.array(predicted_score)

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(label_actual, predicted_score, pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.plot(fpr[0], tpr[0], linestyle="--", color="aqua", label="ROC Curve of Normal Class area = " + str(roc_auc[0]))
    plt.plot(fpr[1], tpr[1], linestyle="--", color="darkorange", label="ROC Curve of Dysplasia Class area = " + str(roc_auc[1]))
    plt.plot([0, 1], [0, 1], 'k--')
    # plt.grid()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Colon Patch Classification ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive rate')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(save_path, "roc_curve.png"))
    plt.close()

def get_confusion_matrix(save_path, data):
    temp = np.array(data.tolist())
    cm_df = pd.DataFrame(temp, range(2), range(2))
    print(cm_df)
    sns.heatmap(cm_df, annot=True, cmap="Blues", fmt=".1f")
    plt.title("Graph VIT for WSI Colon Classification")
    plt.ylabel("Predicted Classes")
    plt.xlabel("Actual Classes")
    plt.savefig(os.path.join(save_path, "confusion_matrix.png"))
    plt.close()


def get_loss_curve(save_path, train_losses, valid_losses):
    if not os.path.isdir(save_path): os.mkdir(save_path)
    plt.plot(train_losses, color="blue", label="train")
    plt.plot(valid_losses, color="red", label="valid")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "loss_curve.png"))
    plt.close()

def get_accuracy_curve(save_path, train_acc, valid_acc):
    if not os.path.isdir(save_path): os.mkdir(save_path)
    plt.plot(train_acc, color="blue", label="train")
    plt.plot(valid_acc, color="red", label="valid")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, "accuracy_curve.png"))
    plt.close()