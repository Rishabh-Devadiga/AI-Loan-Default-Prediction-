import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_feature_importance(importance, names, model_type):
    feature_importance = np.array(importance)
    feature_names = np.array(names)
    fi_df = pd.DataFrame(
        {"Feature Names": feature_names, "Feature Importance": feature_importance}
    ).sort_values(by="Feature Importance", ascending=False)

    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df["Feature Importance"], y=fi_df["Feature Names"])
    plt.title(f"Feature Importance for {model_type}")
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.show()


def display_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def plot_roc_curve(fpr, tpr, model_name):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve - {model_name}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic (ROC) Curve")
    plt.legend(loc="lower right")
    plt.show()
