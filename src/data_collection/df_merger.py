import os
import fnmatch
import pandas as pd
def merge_dfs(directory, file_pattern="*reviews*.csv", out_path = "merged_reviews.csv"):
    df_merge = pd.DataFrame()
    for file in os.listdir(directory):
        if fnmatch.fnmatch(file, file_pattern):
            df_temp = pd.read_csv(os.path.join(directory,file))
            df_temp = df_temp[["text","rating"]]
            df_merge = pd.concat([df_temp,df_merge])
    print(f"Length of concatenated Dataframe before droping duplicates: {len(df_merge)}")
    df_merge.drop_duplicates(subset="text", inplace=True)
    print(f"Length of Dataframe after droping duplicates: {len(df_merge)}")
    n_unique  = df_merge["text"].nunique()
    print(f"Number of unique values in text column: {n_unique} \n This should be the same value as the length of the Dataframe.")
    
    df_merge.to_csv(out_path)

    return merge_dfs


if __name__ == "__main__":
    directory = os.path.join("..", "..","data")
    file_pattern = "*reviews*.csv"
    out_path = os.path.join("..", "..","data", "merged_reviews.csv" )
    merge_dfs(directory, file_pattern,out_path )

