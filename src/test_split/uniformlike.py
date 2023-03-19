import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

def uniform_dist_selection(df_regions: pd.DataFrame, ratio:float=0,mode="test"):
    reference_distribution = np.ones(len(df_regions.columns))/ len(df_regions.columns)

    if ratio != 0:
        n_samples = np.sum(np.random.rand(len(df_regions)) <= ratio)
        
        df_regions_out = df_regions.copy()
        df_regions_out.insert(
                    0, 
                    "JS_div",
                    df_regions.apply(lambda x: jensenshannon(reference_distribution, x), axis=1)
                    )
        indx_test = df_regions_out.sort_values("JS_div", ascending=False)[:n_samples].index

        if mode =="test":
            df_regions_out.insert(0, "test", np.zeros(len(df_regions_out), dtype=bool))
            df_regions_out.loc[indx_test,"test"] = True
        elif mode=="train":
            df_regions_out.insert(0, "test", np.ones(len(df_regions_out), dtype=bool))
            df_regions_out.loc[indx_test,"test"] = False
        return df_regions_out


if __name__ == "__main__":
    folder_ = "./"
    continent_ = "eu"
    
    info_sum_f = pd.read_csv(f"{folder_}/metadata_dates/{continent_}/region_label_freq.csv", index_col=0)
    
    ratio_ = 0.2
    uniform_dist_selection(info_sum_f, ratio=ratio_, mode="test").to_csv(f"{folder_}/split_samples/{continent_}/uniform_{int(ratio_*100)}p_test.csv")
    
    ratio_ = 0.8
    uniform_dist_selection(info_sum_f, ratio=ratio_, mode="train").to_csv(f"{folder_}/split_samples/{continent_}/uniform_{int(ratio_*100)}p_train.csv")

    print(f"Finished and saved into: {folder_}/split_samples/{continent_}/")