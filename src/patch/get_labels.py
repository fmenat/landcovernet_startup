import numpy as np
import pandas as pd
from tqdm import tqdm

from src.core.image_read import LandCoverBuilder

def create_pandas_auxiliar(targets_info_, label_names_):
    info_summary = pd.DataFrame(targets_info_, columns=list(label_names_.keys()), index=ids)
    info_summary.columns = list(label_names_.values())
    info_summary.index.name = "identifier"
    info_summary.replace(np.nan, 0, inplace=True )
    return info_summary


if __name__ == "__main__":
    folder_ = "/ds/images/AI4EO/multi/landcovernet/"
    output_dir = f"{folder_}/metadata_dates""
    continent = "eu"

    LC_build = LandCoverBuilder(f"{folder_}/ref_landcovernet_{continent}_v1")
    label_names_ = LC_build.get_labels_name()
    print("Dataset contains ",len(LC_build), " patches, where skipping equals to ", LC_build.skip_labels,"and labels are ",label_names_.values())

    ids = []
    targets_info_ = []
    targets_info_conf_ = []
    for i in tqdm(range(len(LC_build)),total = len(LC_build)):
        patch_i_label = LC_build.get_label(i)
        labels_array_i = patch_i_label["labels_array"].values[0].flatten()
        labels_array_i_conf = patch_i_label["labels_array"].values[1].flatten()

        unique, counts = np.unique(labels_array_i, return_counts=True)
        info_ = dict(zip(unique,counts))
        info_conf_ = dict(zip(unique,[labels_array_i_conf[np.where(u == labels_array_i)[0]].mean() for u in unique] ))

        targets_info_.append(info_)
        targets_info_conf_.append(info_conf_)
        ids.append(patch_i_label["identifier"])
        
    info_summary = create_pandas_auxiliar(targets_info_, label_names_)
    info_summary.astype(int).to_csv(f"{output_dir}/{continent}/region_label_freq.csv")
    info_summary = create_pandas_auxiliar(targets_info_conf_, label_names_)
    info_summary.to_csv(f"{output_dir}/{continent}/region_label_conf_avg.csv")

    print(f"Finish and stored into {output_dir}/{continent}/region_*.csv")