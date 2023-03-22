from tqdm import tqdm
import pandas as pd
import pathlib
from country_bounding_boxes import countries #https://github.com/graydon/country-bounding-boxes
from shapely.geometry import Polygon
import os

from src.core.image_read import LandCoverBuilder

def intersects(box1, box2):
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1])

def searching_overlap_countries(bbox):
    res = []
    for c in countries:
        if intersects(list(c.bbox), bbox):
            res.append([c.region_un, c.subregion, c.admin])
    return pd.DataFrame(res ,columns=["Region","Subregion", "Country"])    


if __name__ == "__main__":
    #usually executed after having region label metadata
    
    folder_ = "/ds/images/AI4EO/multi/landcovernet/"
    metadata_folder_ = "./metadata"
    output_dir = "./split_samples"
    continent = "sa"

    LC_build = LandCoverBuilder(f"{folder_}/ref_landcovernet_{continent}_v1")
    countries_id = {}
    for i in tqdm(range(len(LC_build)),total = len(LC_build)):
        data_i = LC_build.get_label(i)
        countries_i = searching_overlap_countries(data_i["bbox"])
        countries_id[data_i["identifier"]] = countries_i

    info_countries = pd.concat(countries_id).reset_index().set_index("level_0").drop("level_1", axis=1)
    info_countries[["Region","Subregion","Country"]].to_csv(f"{metadata_folder_}/{continent}/info_countries.csv")
    print(f"Finished and saved into: {metadata_folder_}/{continent}/info_countries.csv")

    if os.path.isfile(f"{metadata_folder_}/{continent}/region_label_freq.csv"):
        info_sum_f_rel = pd.read_csv(f"{metadata_folder_}/{continent}/region_label_freq.csv", index_col=0)
        pd.merge(info_countries, info_sum_f_rel,right_index=True,left_index=True ).to_csv(f"{output_dir}/{continent}/info_countries.csv")
        print(f"Finished and saved into: {output_dir}/{continent}/info_countries.csv")
