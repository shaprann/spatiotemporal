from torch.utils.data import Dataset
import xarray as xr
from tqdm import tqdm
import torch
from functools import partial


class CTGANTorchDataset(Dataset):

    def __init__(self, dataset_manager, device):

        self.device = device
        self.samples = []
        for datatree_node in tqdm(dataset_manager.data.leaves, desc="Converting dataset to training samples"):
            for sample in self.xr_dataset_to_samples(datatree_node.ds):
                self.samples.append(sample)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    @classmethod
    def xr_dataset_to_samples(cls, dataset):

        dataset = (xr.Dataset)(dataset).drop_vars("S1")

        # switch to image coordinates instead of lat, lon
        dataset = dataset.swap_dims({"lat": "y", "lon": "x"})

        dataset["S2_t-1"] = dataset["S2"].shift(timestep=+1)
        dataset["S2_t-2"] = dataset["S2"].shift(timestep=+2)
        dataset["S2_t-3"] = dataset["S2"].shift(timestep=+3)

        dataset["S2_cloud_map_t-1"] = dataset["S2_cloud_map"].shift(timestep=+1)
        dataset["S2_cloud_map_t-2"] = dataset["S2_cloud_map"].shift(timestep=-2)
        dataset["S2_cloud_map_t-3"] = dataset["S2_cloud_map"].shift(timestep=-3)

        dataset = dataset.isel(timestep=slice(3, 30))

        dataset = dataset.drop_dims("polarization")

        # split by timestep using groupby(), remove timestep from resulting tuple
        return [ds for timestep, ds in list(dataset.groupby("timestep"))]

    def get_collate_fn(self):
        return partial(self.collate_fn, device=self.device)

    @classmethod
    def collate_fn(cls, list_of_samples: list, device):
        data = xr.concat(list_of_samples, dim="batch")
        data = data.compute()
        return (
            torch.from_numpy(data["S2"].values.astype(float)).to(device),
            torch.from_numpy(data["S2_t-1"].values.astype(float)).to(device),
            torch.from_numpy(data["S2_t-2"].values.astype(float)).to(device),
            torch.from_numpy(data["S2_t-3"].values.astype(float)).to(device),
            torch.from_numpy(data["S2_cloud_map"].values.astype(float)).to(device),
            torch.from_numpy(data["S2_cloud_map_t-1"].values.astype(float)).to(device),
            torch.from_numpy(data["S2_cloud_map_t-2"].values.astype(float)).to(device),
            torch.from_numpy(data["S2_cloud_map_t-3"].values.astype(float)).to(device),
            data["S2"]
        )
