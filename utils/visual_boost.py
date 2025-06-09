import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def draw_adaboost_accuracy_comparison():
    files = {
        "Linear": "./experiment/boosting/Linear.csv",
        "MLP": "./experiment/boosting/NN.csv",
        "CNN": "./experiment/boosting/CNN.csv"
    }

    colors = {
        "Linear": "tab:blue",
        "MLP": "tab:green",
        "CNN": "tab:red"
    }

    plt.figure(figsize=(8, 5))
    max_len = 0
    acc_dict = {}

    # Load and collect accuracy data
    for label, path in files.items():
        df = pd.read_csv(path)
        test_acc = df['test_accuracy'].values * 100
        train_acc = df['train_accuracy'].values * 100
        acc_dict[label] = {
            "test": test_acc,
            "train": train_acc
        }
        max_len = max(max_len, len(test_acc))

    # Plot each model
    for label, accs in acc_dict.items():
        color = colors[label]

        # Test accuracy (solid)
        x_test = np.arange(len(accs["test"]))
        plt.plot(x_test, accs["test"], label=f"{label} (Test)", linewidth=2, color=color)

        # Train accuracy (solid, alpha low)
        x_train = np.arange(len(accs["train"]))
        plt.plot(x_train, accs["train"], label=f"{label} (Train)", linewidth=2, color=color, alpha=0.4)

        # Test accuracy supplement (dashed)
        if len(accs["test"]) < max_len:
            last_val = accs["test"][-1]
            x_fill = np.arange(len(accs["test"]) - 1, max_len)
            y_fill = np.full_like(x_fill, last_val, dtype=float)
            plt.plot(x_fill, y_fill, linestyle='--', linewidth=2, alpha=1, color=color)

        # Train accuracy supplement (dashed, faded)
        if len(accs["train"]) < max_len:
            last_val = accs["train"][-1]
            x_fill = np.arange(len(accs["train"]) - 1, max_len)
            y_fill = np.full_like(x_fill, last_val, dtype=float)
            plt.plot(x_fill, y_fill, linestyle='--', linewidth=1.5, alpha=0.4, color=color)

    plt.xlabel("Number of Weak Classifiers", fontsize=12)
    plt.ylabel("Accuracy (%)", fontsize=12)
    plt.title("Training and Test Accuracy vs Boosting Rounds", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("./image/boost_acc_comparison.pdf")
    plt.show()


def draw_adaboost_loss_comparison():
    files = {
        "Linear": "./experiment/boosting/Linear.csv",
        "MLP": "./experiment/boosting/NN.csv",
        "CNN": "./experiment/boosting/CNN.csv"
    }

    colors = {
        "Linear": "tab:blue",
        "MLP": "tab:green",
        "CNN": "tab:red"
    }

    plt.figure(figsize=(8, 5))
    max_len = 0
    loss_dict = {}

    # Load and collect loss data
    for label, path in files.items():
        df = pd.read_csv(path)
        test_loss = df['test_loss'].values
        train_loss = df['train_loss'].values
        loss_dict[label] = {
            "test": test_loss,
            "train": train_loss
        }
        max_len = max(max_len, len(test_loss))

    # Plot each model
    for label, losses in loss_dict.items():
        color = colors[label]

        # Test loss (solid)
        x_test = np.arange(len(losses["test"]))
        plt.plot(x_test, losses["test"], label=f"{label} (Test)", linewidth=2, color=color)

        # Train loss (solid, alpha low)
        x_train = np.arange(len(losses["train"]))
        plt.plot(x_train, losses["train"], label=f"{label} (Train)", linewidth=2, color=color, alpha=0.4)

        # Test loss supplement (dashed)
        if len(losses["test"]) < max_len:
            last_val = losses["test"][-1]
            x_fill = np.arange(len(losses["test"]) - 1, max_len)
            y_fill = np.full_like(x_fill, last_val, dtype=float)
            plt.plot(x_fill, y_fill, linestyle='--', linewidth=1.5, alpha=0.7, color=color)

        # Train loss supplement (dashed, faded)
        if len(losses["train"]) < max_len:
            last_val = losses["train"][-1]
            x_fill = np.arange(len(losses["train"]) - 1, max_len)
            y_fill = np.full_like(x_fill, last_val, dtype=float)
            plt.plot(x_fill, y_fill, linestyle='--', linewidth=1.5, alpha=0.4, color=color)

    plt.xlabel("Number of Weak Classifiers", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training and Test Loss vs Boosting Rounds", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("./image/boost_loss_comparison.pdf")
    plt.show()


def draw_adaboost_loss_comparison_logscale():
    files = {
        "Linear": "./experiment/boosting/Linear.csv",
        "MLP": "./experiment/boosting/NN.csv",
        "CNN": "./experiment/boosting/CNN.csv"
    }

    colors = {
        "Linear": "tab:blue",
        "MLP": "tab:green",
        "CNN": "tab:red"
    }

    plt.figure(figsize=(8, 5))
    max_len = 0
    loss_dict = {}

    for label, path in files.items():
        df = pd.read_csv(path)
        test_loss = df['test_loss'].values
        train_loss = df['train_loss'].values
        loss_dict[label] = {
            "test": test_loss,
            "train": train_loss
        }
        max_len = max(max_len, len(test_loss))

    for label, losses in loss_dict.items():
        color = colors[label]

        # Test Loss (solid)
        x_test = np.arange(len(losses["test"]))
        plt.plot(x_test, losses["test"], label=f"{label} (Test)", linewidth=2, color=color)

        # Train Loss (solid, lighter)
        x_train = np.arange(len(losses["train"]))
        plt.plot(x_train, losses["train"], label=f"{label} (Train)", linewidth=2, color=color, alpha=0.4)

        # Supplement dashed lines
        if len(losses["test"]) < max_len:
            last_val = losses["test"][-1]
            x_fill = np.arange(len(losses["test"]) - 1, max_len)
            y_fill = np.full_like(x_fill, last_val, dtype=float)
            plt.plot(x_fill, y_fill, linestyle='--', linewidth=1.5, alpha=0.7, color=color)

        if len(losses["train"]) < max_len:
            last_val = losses["train"][-1]
            x_fill = np.arange(len(losses["train"]) - 1, max_len)
            y_fill = np.full_like(x_fill, last_val, dtype=float)
            plt.plot(x_fill, y_fill, linestyle='--', linewidth=1.5, alpha=0.4, color=color)

    plt.xlabel("Number of Weak Classifiers", fontsize=12)
    plt.ylabel("Loss (Log Scale)", fontsize=12)
    plt.title("Training and Test Loss (Log Scale)", fontsize=13)
    plt.yscale("log")  # <<< Log scale added here
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    plt.savefig("./image/boost_loss_comparison_logscale.pdf")
    plt.show()


if __name__ == "__main__":
    draw_adaboost_loss_comparison_logscale()