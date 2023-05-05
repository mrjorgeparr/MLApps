import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_class_distribution(plot_path, df):
        df["labels"].hist(bins=10)
        plt.title("Distribution of the Ratings in Dataset")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.savefig(plot_path)

if __name__ == "__main__":
        data_path = os.path.join("..", "..", "data", "preprocessed_data.csv")
        plot_path = os.path.join("..", "..", "figures", "Data_distribution.png")
        df = pd.read_csv(data_path)
        plot_class_distribution(plot_path,df)
