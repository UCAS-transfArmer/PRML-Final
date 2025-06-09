import pandas as pd
import matplotlib.pyplot as plt

def ema(series, alpha=0.15):
    return series.ewm(alpha=alpha).mean()

# NOTE: the min and max are the same. I don't want to change it.

def draw_loss():
    # Load CSVs
    train_loss = pd.read_csv("./experiment/logistic/train_loss.csv")
    test_loss = pd.read_csv("./experiment/logistic/test_loss.csv")

    # Extract training loss info
    step_train = train_loss["Step"]
    train_mean = train_loss.iloc[:, 1]
    train_min = train_loss.iloc[:, 2]
    train_max = train_loss.iloc[:, 3]

    # Extract test loss info
    step_test = test_loss["Step"]
    test_mean = test_loss.iloc[:, 1]
    test_min = test_loss.iloc[:, 2]
    test_max = test_loss.iloc[:, 3]

    # Plot figure
    plt.figure(figsize=(8, 5))

    # Train loss area + main + EMA
    plt.fill_between(step_train, train_min, train_max, color="lightblue", alpha=0.4)
    plt.plot(step_train, train_mean, label="Train Loss", color=(0.678, 0.847, 0.902), linewidth=1)
    plt.plot(step_train, ema(train_mean), label="Train EMA", color="tab:blue", linewidth=2)

    # Test loss area + main + EMA
    plt.fill_between(step_test, test_min, test_max, color="moccasin", alpha=0.4)
    plt.plot(step_test, test_mean, label="Test Loss", color="tab:pink", linewidth=1, alpha=0.5)
    plt.plot(step_test, ema(test_mean), label="Test EMA", color="tab:red", linewidth=2)

    # Labels and style
    plt.xlabel("Training Step", fontsize=12)
    plt.ylabel("Cross-Entropy Loss", fontsize=12)
    plt.title("Training and Test Loss over Time", fontsize=13)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(fontsize=11)
    plt.tight_layout()

    # Save figure
    plt.savefig("./image/logistic_loss_curve.pdf")
    plt.show()


def draw_acc():
    # Load data
    test_acc = pd.read_csv("./experiment/logistic/test_acc.csv")
    train_acc = pd.read_csv("./experiment/logistic/train_acc.csv")

    # Extract columns
    test_step = test_acc["Step"]
    test_mean = test_acc[test_acc.columns[1]]
    test_min = test_acc[test_acc.columns[2]]
    test_max = test_acc[test_acc.columns[3]]

    train_step = train_acc["Step"]
    train_mean = train_acc[train_acc.columns[1]] * 100
    train_min = train_acc[train_acc.columns[2]] * 100
    train_max = train_acc[train_acc.columns[3]] * 100

    # Create figure
    plt.figure(figsize=(8, 5))

    # Plot train accuracy
    plt.fill_between(train_step, train_min, train_max, color="lightgray", alpha=0.5)
    plt.plot(train_step, train_mean, label="Train Accuracy", color=(0.678, 0.847, 0.902), linewidth=1)
    plt.plot(train_step, ema(train_mean), label="Train EMA", color="tab:blue", linewidth=2)

    # Plot test accuracy
    plt.fill_between(test_step, test_min, test_max, color="lightgray", alpha=0.3)
    plt.plot(test_step, test_mean, label="Test Accuracy", color="tab:pink", linewidth=1, alpha=0.5)
    plt.plot(test_step, ema(test_mean), label="Test EMA", color="tab:red", linewidth=2)

    # Labels and formatting
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Test Accuracy over Time (Logistic Regression)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save and show
    plt.savefig("./image/logistic_acc_curve.pdf")
    plt.show()

if __name__ == "__main__":
    draw_acc()