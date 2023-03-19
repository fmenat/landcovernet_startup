import xarray as xr
import os

from .image_read import LandCoverBuilder

if __name__ == "__main__":
    folder_ = "/ds/images/AI4EO/multi/landcovernet/"
    continent = "eu"


    #into big dataset
    LC_build = LandCoverBuilder(f"{folder_}/ref_landcovernet_{continent}_v1")
    print("Dataset contains ",len(LC_build), " patches, where skipping equsl to ", LC_build.skip_labels)
    ids = []
    targets = []
    view_names = LC_build.view_names
    views = {v:[] for v in view_names}
    for i in range(len(LC_build)):
        identifier = LC_build.get_id(i)
        if not os.path.isfile(f"{folder_}/netcdf/patches/{identifier}.nc"):
            data_i = LC_build[i]
            ids.append(data_i["id"])
            targets.append(data_i["target"][0])
            for v in data_i["view_names"]:
                views[v].append(data_i["views"][v])

            #storage
            data_store = xr.Dataset(data_vars={"target": data_i["target"][0], **data_i["views"]})
            data_store = data_store.assign_attrs({"view_names": view_names,"id": data_i["id"]})
            data_store.to_netcdf(f"{folder_}/netcdf/{continent}/patches/{data_i['id']}.nc", engine="h5netcdf")
        else:
            print(f"Skipping file {identifier} because created")

    print("Finish open data")

    targets = xr.concat(targets, dim="identifier")
    for v in view_names:
        views[v] = xr.concat(views[v], dim="identifier")

    print("Creating dataset and into storage")
    data_store = xr.Dataset(data_vars={"target": targets, **views})
    data_store = data_store.assign_attrs({"view_names": view_names, "target_names": LC_build.label_to_name})
    data_store["ids"] = xr.DataArray(ids, dims="identifier", coords={"identifier": np.arange(len(ids))})
    data_store.to_netcdf(f"{folder_}/netcdf/{continent}/all.nc", engine="h5netcdf")
