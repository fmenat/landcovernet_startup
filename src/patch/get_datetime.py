import numpy as np
import pandas as pd
from tqdm import tqdm
import os
from pathlib import Path

from src.core.image_read import LandCoverBuilder

if __name__ == "__main__":
    folder_ = "/ds/images/AI4EO/multi/landcovernet/"
    output_dir = f"{folder_}/metadata_dates"
    continent = "eu"
    
    LC_build = LandCoverBuilder(f"{folder_}/ref_landcovernet_{continent}_v1")
    print("Dataset contains ",len(LC_build), " patches, where skipping equals to ", LC_build.skip_labels)
    ids = []
    targets = []
    view_names = LC_build.view_names
    views = {v:[] for v in view_names}
    df_meta_viewn_patches = {v:[] for v in view_names}
    for i in tqdm(range(len(LC_build)),total = len(LC_build)):
        patch_i = LC_build[i]
        
        identifier = patch_i["id"]
        Path(f"{output_dir}/{continent}/{identifier}").mkdir(parents=True, exist_ok=True)

        for view_n in patch_i["view_names"]: 
            #if not os.path.isfile(f"{output_dir}/{continent}/{identifier}/{view_n}.csv"):
            df_meta_viewn = pd.DataFrame(patch_i["views_date"][view_n], columns=["dates"])
            df_meta_viewn["full_name"] = [f"ref_landcovernet_{continent}_v1_source_{view_n}_{identifier}_{v}" for v in df_meta_viewn["dates"]]
            df_meta_viewn["full_path"] = [v.parent for v in patch_i["views_info"][view_n]]
            df_meta_viewn["year"] = df_meta_viewn["dates"].apply(lambda x: int(x[:4]))
            df_meta_viewn["month"] = df_meta_viewn["dates"].apply(lambda x: int(x[4:6]))
            df_meta_viewn["day"] = df_meta_viewn["dates"].apply(lambda x: int(x[6:]))
            df_meta_viewn.to_csv(f"{output_dir}/{continent}/{identifier}/{view_n}.csv",index=False)

            df_meta_viewn["patch_id"] = identifier
            df_meta_viewn_patches[view_n].append(df_meta_viewn)
            #else:
            #    pass
            #    print("Skipping")
            #print("Finishing data ",identifier)

    for view_n in view_names:
        aux_save = pd.concat(df_meta_viewn_patches[view_n], axis=0)
        aux_save = aux_save[["patch_id","dates","year","month","day","full_name","full_path"]]
        aux_save.to_csv(f"{output_dir}/{continent}/{view_n}_overall_info.csv",index=False)
        print(f"Finish and stored {view_n} into {output_dir}/{continent}/{view_n}_overall_info.csv")

    print("Finish datetime calculation")