import matplotlib.pyplot as plt

def plot_series(df, col: str, title: str = "ADT over Time"):
    plt.figure(figsize=(10, 3.5))
    plt.plot(df.index, df[col], linewidth=1.2)
    plt.title(title)
    plt.xlabel("Year")
    plt.ylabel(col)
    plt.tight_layout()
    plt.show()

def plot_union(df_union, ylabel: str = "ADT"):
    plt.figure(figsize=(14, 4.5))
    if "x_train" in df_union: plt.plot(df_union.index, df_union["x_train"], linewidth=1.2, linestyle="-", label="Training Data")
    if "y_test" in df_union: plt.plot(df_union.index, df_union["y_test"], linewidth=1.2, linestyle="-", label="Test Data")
    if "y_pred" in df_union: plt.plot(df_union.index, df_union["y_pred"], linewidth=1.2, linestyle="--", label="Test Prediction")
    if "residuals" in df_union: plt.plot(df_union.index, df_union["residuals"], linewidth=1.0, linestyle=":", label="Residuals")
    plt.xlabel("Year"); plt.ylabel(ylabel)
    plt.legend(loc="best"); plt.grid(axis="x"); plt.tight_layout(); plt.show()

def plot_test_vs_pred(valid_df, ylabel: str = "ADT"):
    plt.figure(figsize=(10, 4.5))
    plt.plot(valid_df.index, valid_df["y_test"], linewidth=1.2, linestyle="-", label="Test Data")
    plt.plot(valid_df.index, valid_df["y_pred"], linewidth=1.2, linestyle="--", label="Test Prediction")
    plt.xlabel("Year"); plt.ylabel(ylabel)
    plt.legend(loc="best"); plt.grid(axis="x"); plt.xticks(rotation=30); plt.tight_layout(); plt.show()

def plot_loss(history):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history.get("loss", []), linestyle="-", label="Training")
    plt.plot(history.history.get("val_loss", []), linestyle="-", label="Validation")
    plt.ylabel("Loss"); plt.xlabel("Epoch"); plt.legend(loc="best"); plt.tight_layout(); plt.show()
