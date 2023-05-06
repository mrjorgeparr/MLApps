import os
import pandas as pd
import matplotlib.pyplot as plt

#def result_evaluation(csv_file):

def plot_results(df, plot_path):
    df = df.sort_values(by=['f1'])
    df['Model_combination'] = df['Classifier'] + " " + df['Vectorizer']
    df = df.set_index('Model_combination')
    metrics = ["accuracy","f1","precision","recall"]
    df[metrics].plot(kind='bar', figsize=(20,15))

    plt.title('Performance Comparison')
    plt.xlabel('Model Combination')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Metric Scores')
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(plot_path, bbox_inches='tight')

if __name__ == "__main__":
    df_all = pd.read_csv(os.path.join("..","..","reports","all_classes_results.csv"))
    df_four = pd.read_csv(os.path.join("..","..","reports","four_classes_results.csv"))
    df_two = pd.read_csv(os.path.join("..","..","reports","two_classes_results.csv"))

    all_path = os.path.join("..", "..", "figures", "all_results_all_classes.png")
    four_path = os.path.join("..", "..", "figures", "all_results_four_classes.png")
    two_path = os.path.join("..", "..", "figures", "all_results_two_classes.png")
    plot_results(df_all, all_path)
    plot_results(df_four, four_path)
    plot_results(df_two, two_path)