import pandas as pd
import numpy as np

def random_selection(df_regions: pd.DataFrame, ratio:float=0, n_samples: int = 0):
    #ratio of test samples (Between 0-1)
    #n_samples of test samples (between 0-len(samples))
    mask_selected = np.zeros(len(df_regions), dtype=bool) 
    if ratio != 0:
        mask_selected = np.random.rand(len(df_regions)) <= ratio
    elif n_samples != 0:
        mask_selected[np.random.choice(np.arange(len(df_regions)), replace=False, size= 10)] = True
    
    if np.sum(mask_selected) != 0:
        df_regions_out = df_regions.copy()
        if "test" in df_regions_out.columns:
            df_regions_out.drop("test", axis=1, inplace=True)
        df_regions_out.insert(0, "test", mask_selected)
        return df_regions_out


if __name__ == "__main__":
    folder_ = "./"
    continent_ = "eu"
    ratio_ = 0.2

    info_sum_f = pd.read_csv(f"{folder_}/metadata_dates/{continent_}/region_label_freq.csv", index_col=0)
    
    file_name = f"{folder_}/split_samples/{continent_}/random_{int(ratio_*100)}p.csv"
    random_selection(info_sum_f, ratio=ratio_).to_csv(file_name)

    print(f"Finished and saved into: {file_name}")