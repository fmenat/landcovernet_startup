import numpy as np
import pandas as pd

def countries_selection(df_regions: pd.DataFrame, countries:list=[],mode="test"):
    if len(countries)!= 0:
        indx_sel_total = []
        for c in countries:
            indx_sel_total.extend(df_regions[df_regions["Country"] == c].index)

        df_regions_out = df_regions.copy()
        if mode =="test":
            df_regions_out.insert(0, "test", np.zeros(len(df_regions_out), dtype=bool))
            df_regions_out.loc[indx_sel_total,"test"] = True
        elif mode=="train":
            df_regions_out.insert(0, "test", np.ones(len(df_regions_out), dtype=bool))
            df_regions_out.loc[indx_sel_total,"test"] = False

        
        return df_regions_out



if __name__ == "__main__":
    folder_ = "./"
    continent_ = "eu"
    countries=["Germany"]
    
    info_countries = pd.read_csv(f"{folder_}/metadata/{continent_}/info_countries.csv", index_col=0)
    
    info_countries_ext = countries_selection(info_countries, countries=countries, mode="test")
    info_countries_ext = info_countries_ext[~info_countries_ext.index.duplicated(keep='first')]
    
    countries_str = "-".join(countries)
    file_name = f"{folder_}/split_samples/{continent_}/countries_{countries_str}_test.csv"
    info_countries_ext.to_csv(file_name)
    print(f"Finished and saved into: {file_name}")