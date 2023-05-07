import pandas as pd
import os

def map_to_four(rating):
    if 1.0 <= rating <= 2.0:
        return 1
    elif 3.0 <= rating <= 5.0:
        return 2
    elif 6.0 <= rating <= 8.0:
        return 3
    elif 9.0 <= rating <= 10.0:
        return 4
    
def map_to_two(rating):
    """
    Maps the classes to only two classes
    """

    if 1.0 <= rating <= 5.0:
        return 1
    elif 6.0 <= rating <= 10.0:
        return 2
    
if __name__ == "__main__":
    df = pd.read_csv(os.path.join("..", "..", "data", "preprocessed_data.csv"))
    df_four = df.copy()
    df_four["labels"] = df_four["labels"].apply(map_to_four)
    df_two = df.copy()
    df_two["labels"] = df_two["labels"].apply(map_to_two)

    df_four.to_csv(os.path.join("..", "..", "data", "preprocessed_data_four.csv"))
    df_two.to_csv(os.path.join("..", "..", "data", "preprocessed_data_two.csv"))