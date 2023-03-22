import json, os
from pathlib import Path
import pandas as pd
import xarray as xr
import numpy as np

def open_rio(file):
    try:
        import rioxarray
        return rioxarray.open_rasterio(file)
    except:
        return xr.open_rasterio(file)

class LandCoverBuilder(object):
    def __init__(self, download_folder="./", return_torch: bool = False):
        self.download_folder = Path(download_folder)
        self.return_torch = return_torch

        with open( self.download_folder / "catalog.json", "rb") as v:
            self.global_catalog = json.load(v)
            for v in self.global_catalog["links"]:
                if v["rel"] != "child":
                    continue
                elif "labels" in v["href"]:
                    labelsjson_path =  self.download_folder / v["href"]
            with open( labelsjson_path, "rb") as v: #open labels json
                self.labels_catalog = json.load(v)
                self.labels_path = self.download_folder / self.labels_catalog["id"]
            
        for i, v in enumerate(self.labels_catalog["links"]):
            if v["rel"] == "item":
                self.skip_labels = i
                break
        
        self.label_to_name = self.get_labels_name()
        self.view_names = ["sentinel_1", "sentinel_2", "landsat_8"]
        self.views_bands = {"landsat_8": [f"B{v:02d}" for v in range(1,7+1)],
                            "sentinel_1": ["VH", "VV"],
                            "sentinel_2": [f"B{v:02d}" for v in range(1,9+1)] + ["B8A","B11","B12"] + ["CLD"]+["SCL"]
                           }
    
    def get_labels_name(self):
        label_info_path = self.labels_catalog["links"][self.skip_labels:][0]["href"]
        with open(self.labels_path / label_info_path, "rb") as v: #open labels json
            label_info_indx = json.load(v)
            return {a["value"][0]:a["summary"] for a in label_info_indx["assets"]["labels"]["file:values"]}

    def get_id(self, indx):
        label_info_path = self.labels_catalog["links"][self.skip_labels:][indx]["href"]
        with open(self.labels_path / label_info_path, "rb") as f: #open labels json
            label_info_indx = json.load(f)
            tile_id, chip_id = label_info_indx["id"].split("_")[-2:]
            return  tile_id+"_"+chip_id

    def get_label(self, indx):
        label_info_path = self.labels_catalog["links"][self.skip_labels:][indx]["href"]
        label_path_child = label_info_path[::-1].split("/", 1)[1][::-1]
        
        #get identifier
        with open(self.labels_path / label_info_path, "rb") as f: #open labels json
            label_info_indx = json.load(f)
            tile_id, chip_id = label_info_indx["id"].split("_")[-2:]
            identifier = tile_id+"_"+chip_id
            geometry = label_info_indx["geometry"]
            bbox = label_info_indx["bbox"]

        labels_array = open_rio(self.labels_path / label_path_child / "labels.tif")
        return dict(
            labels_array=labels_array,
            label_info_indx=label_info_indx,
            identifier=identifier,
            geometry=geometry,
            bbox=bbox
        )
    
    def get_dates(self, indx):
        dict_labels = self.get_label(indx)   
            
        views_info = {v: [] for v in self.view_names} 
        for v in dict_labels["label_info_indx"]["links"]:
            if v["rel"] != "source":
                continue
            if "source_sentinel_1" in  v["href"]:
                views_info["sentinel_1"].append( self.download_folder /  v["href"].split("../")[-1])
            elif "source_sentinel_2" in  v["href"]:
                views_info["sentinel_2"].append( self.download_folder /  v["href"].split("../")[-1])
            elif "source_landsat_8" in  v["href"]:
                views_info["landsat_8"].append( self.download_folder /  v["href"].split("../")[-1])
            
        views_dates = {v: [] for v in self.view_names}
        for view_name in self.view_names:
            views_p_times,times_info  = [], []
            for folder_time in views_info[view_name]:
                times_info.append(str(folder_time.parent).split("_")[-1])
            views_dates[view_name] = times_info
            
        return {
                "id":dict_labels["identifier"],
                    "bbox": dict_labels["bbox"],
                    "geometry": dict_labels["geometry"],
                    "label_info": dict_labels["label_info_indx"], 
                    "views_info": views_info, 
                    "views_date": views_dates,
                    "view_names": self.view_names
                    }
    
    def __len__(self):
        return len(self.labels_catalog["links"][self.skip_labels:])
        
    def __getitem__(self,indx):
        #perhaps too much read disk     
        dict_labels = self.get_label(indx)   
            
        views_info = {v: [] for v in self.view_names} 
        for v in dict_labels["label_info_indx"]["links"]:
            if v["rel"] != "source":
                continue
            if "source_sentinel_1" in  v["href"]:
                views_info["sentinel_1"].append( self.download_folder /  v["href"].split("../")[-1])
            elif "source_sentinel_2" in  v["href"]:
                views_info["sentinel_2"].append( self.download_folder /  v["href"].split("../")[-1])
            elif "source_landsat_8" in  v["href"]:
                views_info["landsat_8"].append( self.download_folder /  v["href"].split("../")[-1])
            
        views = {v: [] for v in self.view_names}
        views_dates = views.copy()
        for view_name in self.view_names:
            views_p_times,times_info  = [], []
            for folder_time in views_info[view_name]:
                data_l = xr.concat([open_rio(folder_time.parent / (l_b + ".tif")) for l_b in self.views_bands[view_name]]
                      , dim="band")
                views_p_times.append(data_l)
                times_info.append(str(folder_time.parent).split("_")[-1])
            views[view_name] = xr.concat(views_p_times, pd.Index(times_info, name="time")).assign_coords({"band": self.views_bands[view_name], "x":np.arange(data_l.shape[-2]), "y":np.arange(data_l.shape[-1])}).rename({"band":view_name+"_band"})
            views_dates[view_name] = times_info
            
        additional = {
                    "bbox": dict_labels["bbox"],
                    "geometry": dict_labels["geometry"],
                    "label_info": dict_labels["label_info_indx"], 
                    "views_info": views_info, 
                    "views_date": views_dates,
                    }
        data = {
            "id":dict_labels["identifier"],
               "target": dict_labels["labels_array"].assign_coords({"x":np.arange(data_l.shape[-2]), "y":np.arange(data_l.shape[-1])}).rename({"band":"target_band"}),
               "views": views,
               "view_names": self.view_names
               }
                
        #normalize 
        #fill nans

        if self.return_torch:
            import torch
            data["views"] = {v:  torch.Tensor(data["views"][v].values.astype(np.float32)) for v in self.view_names}
            data["target"] = torch.Tensor(data["target"][v].values.astype(np.int))[0] #first is the label

            
        return dict(data, **additional)