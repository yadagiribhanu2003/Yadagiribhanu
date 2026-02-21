import matplotlib.pyplot as plt
import pandas as pd
import os
import config

def plot_results(csv_path, save_name):

    df = pd.read_csv(csv_path)

    plt.figure()
    plt.plot(df["timesteps"], df["win_rate"])
    plt.xlabel("Timesteps")
    plt.ylabel("Win Rate")
    plt.title("Win Rate Over Training")
    plt.grid()

    os.makedirs(config.PLOT_DIR, exist_ok=True)
    plt.savefig(os.path.join(config.PLOT_DIR, save_name))
    plt.close()
