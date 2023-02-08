import torch
from torch.utils.data import Dataset
import numpy as np

class CTGANTorchDataset(Dataset):

    bands = [3, 2, 1, 7]  # [red, green, blue, NIR]

    def __init__(self, dataset_manager, device):

        self.device = device
        if not dataset_manager.has_cloud_maps:
            raise ValueError("Provided dataset manager does not contain paths to cloud maps. "
                             "Check whether path to cloud maps directory has been provided to dataset manager.")
        self.manager = dataset_manager

        self.data = self.manager.data.copy()
        self.data = self.data.drop("S1", axis=1)

        self.data["S2_t-1"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(1)
        self.data["S2_t-2"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(2)
        self.data["S2_t-3"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(3)
        self.data["S2_cloud_map_t-1"] = self.data["S2_cloud_map"].groupby(level=["ROI", "tile", "patch"]).shift(1)
        self.data["S2_cloud_map_t-2"] = self.data["S2_cloud_map"].groupby(level=["ROI", "tile", "patch"]).shift(2)
        self.data["S2_cloud_map_t-3"] = self.data["S2_cloud_map"].groupby(level=["ROI", "tile", "patch"]).shift(3)

        self.data = self.data.dropna(how="any")

    def __len__(self):
        return len(self.data)

    # TODO: band selection is faulty, fix it
    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        index = sample.name

        original_s2_image = self.manager.read_tif(sample["S2"])

        # need this for cloud maps
        original_input_images = [
            self.manager.read_tif(sample["S2_t-1"]),
            self.manager.read_tif(sample["S2_t-2"]),
            self.manager.read_tif(sample["S2_t-3"])
        ]

        input_cloud_maps = [
            self.manager.get_cloud_map(cloud_map_path=sample["S2_cloud_map_t-1"],
                                       s2_image=self.rescale(self.bands_last(original_input_images[0]))),
            self.manager.get_cloud_map(cloud_map_path=sample["S2_cloud_map_t-2"],
                                       s2_image=self.rescale(self.bands_last(original_input_images[1]))),
            self.manager.get_cloud_map(cloud_map_path=sample["S2_cloud_map_t-3"],
                                       s2_image=self.rescale(self.bands_last(original_input_images[2])))
        ]

        input_images = [
            self.rescale(image[self.bands]) for image in original_input_images
        ]

        target_image = self.rescale(original_s2_image[self.bands])

        cloud_percentage = self.manager.get_cloud_map(
            cloud_map_path=sample["S2_cloud_map"],
            s2_image=self.rescale(self.bands_last(original_s2_image))
        ).mean()

        return {
            "index":                index,
            "original_s2_image":    original_s2_image,
            "target_image":         torch.from_numpy(target_image).to(self.device),
            "inputs":               [
                                        torch.from_numpy(s2_image).to(self.device)
                                        for s2_image in input_images
                                    ],
            "input_cloud_maps":     [
                                        torch.from_numpy(cloud_map).to(self.device)
                                        for cloud_map in input_cloud_maps
                                    ],
            "cloud_percentage":     cloud_percentage
        }

    def collate_fn(self, list_of_samples):

        result = {
            "index": [],
            "original_s2_image": [],
            "target_image": [],
            "inputs": [[], [], []],
            "input_cloud_maps": [[], [], []],
            "cloud_percentage": []
        }
        for sample in list_of_samples:
            result["index"].append(sample["index"])
            result["original_s2_image"].append(sample["original_s2_image"])
            result["target_image"].append(sample["target_image"])
            result["inputs"][0].append(sample["inputs"][0])
            result["inputs"][1].append(sample["inputs"][1])
            result["inputs"][2].append(sample["inputs"][2])
            result["input_cloud_maps"][0].append(sample["input_cloud_maps"][0])
            result["input_cloud_maps"][1].append(sample["input_cloud_maps"][1])
            result["input_cloud_maps"][2].append(sample["input_cloud_maps"][2])
            result["cloud_percentage"].append(sample["cloud_percentage"])

        result["target_image"] = torch.stack(result["target_image"])
        result["inputs"][0] = torch.stack(result["inputs"][0])
        result["inputs"][1] = torch.stack(result["inputs"][1])
        result["inputs"][2] = torch.stack(result["inputs"][2])
        result["input_cloud_maps"][0] = torch.stack(result["input_cloud_maps"][0])
        result["input_cloud_maps"][1] = torch.stack(result["input_cloud_maps"][1])
        result["input_cloud_maps"][2] = torch.stack(result["input_cloud_maps"][2])

        return result

    @staticmethod
    def bands_last(s2_image):
        return np.transpose(s2_image, (1, 2, 0))

    @staticmethod
    def rescale(s2_image):
        return s2_image.clip(0, 10000) / 10000

    @staticmethod
    def rescale_back(s2_image):
        return np.round(s2_image * 10000).astype('uint16')

