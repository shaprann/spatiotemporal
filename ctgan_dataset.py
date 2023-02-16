from abc import ABC

import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
import math


class CTGANTorchDataset(Dataset):

    bands = [3, 2, 1, 7]  # [red, green, blue, NIR]

    def __init__(self, dataset_manager, mode=None):

        self._check_init_arguments(dataset_manager, mode)
        self.manager = dataset_manager
        self.mode = mode
        self.data = self.manager.data_subset(split=mode).copy()
        self._prepare_data()

        # copy some function from manager for better code readability
        self.rescale_s2 = self.manager.utils.rescale_s2
        self.read_tif = self.manager.utils.read_tif
        self.get_cloud_map = self.manager.utils.get_cloud_map

    @staticmethod
    def _check_init_arguments(dataset_manager, mode):

        if not dataset_manager.has_cloud_maps:
            raise ValueError("Provided dataset manager does not contain paths to cloud maps. "
                             "Check whether path to cloud maps directory has been provided to dataset manager.")

        if mode is not None and mode not in ["train", "test", "val"]:
            raise ValueError(f"Incorrect mode provided. "
                             f"Supported modes: None, 'train', 'test', 'val'. Got instead: {mode}")

    def _prepare_data(self):

        self.data = self.data.drop("S1", axis=1)

        self.data["S2_t-1"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(1)
        self.data["S2_t-2"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(2)
        self.data["S2_t-3"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(3)

        self.data["S2CLOUDMAP_t-1"] = self.data["S2CLOUDMAP"].groupby(level=["ROI", "tile", "patch"]).shift(1)
        self.data["S2CLOUDMAP_t-2"] = self.data["S2CLOUDMAP"].groupby(level=["ROI", "tile", "patch"]).shift(2)
        self.data["S2CLOUDMAP_t-3"] = self.data["S2CLOUDMAP"].groupby(level=["ROI", "tile", "patch"]).shift(3)

        self.data = self.data.dropna(how="any")

    def sneak_peek(self, idx):
        """
        Allows to take a sneak peek at cloud percentage at specific index.
        This can reduce computation because batch can be discarded without loading the whole sample

        :param idx: same idx as in __getitem__()
        :return: cloud percentage
        """
        index = self.data.index[idx]
        cloud_map = self.get_cloud_map(index=index)
        return cloud_map.mean()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        index = sample.name

        original_s2_image = self.read_tif(sample["S2"])

        # need this for cloud maps
        original_input_images = [
            self.read_tif(sample["S2_t-1"]),
            self.read_tif(sample["S2_t-2"]),
            self.read_tif(sample["S2_t-3"])
        ]

        input_cloud_maps = [
            self.get_cloud_map(cloud_map_path=sample["S2CLOUDMAP_t-1"], s2_image=original_input_images[0]),
            self.get_cloud_map(cloud_map_path=sample["S2CLOUDMAP_t-2"], s2_image=original_input_images[1]),
            self.get_cloud_map(cloud_map_path=sample["S2CLOUDMAP_t-3"], s2_image=original_input_images[2])
        ]

        input_images = [
            self.rescale_s2(image)[self.bands] for image in original_input_images
        ]

        target_image = self.rescale_s2(original_s2_image)[self.bands]

        cloud_percentage = self.get_cloud_map(
            cloud_map_path=sample["S2CLOUDMAP"],
            s2_image=original_s2_image
        ).mean()

        return {
            "index":                index,
            "original_s2_image":    original_s2_image,
            "target_image":         torch.from_numpy(target_image),
            "inputs":               [
                                        torch.from_numpy(s2_image)
                                        for s2_image in input_images
                                    ],
            "input_cloud_maps":     [
                                        torch.from_numpy(cloud_map)
                                        for cloud_map in input_cloud_maps
                                    ],
            "cloud_percentage":     cloud_percentage
        }

    @staticmethod
    def collate_fn(list_of_samples):

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


class S1CTGANTorchDataset(Dataset):

    bands = [3, 2, 1, 7]  # [red, green, blue, NIR]

    def __init__(self, dataset_manager, mode=None):

        self._check_init_arguments(dataset_manager, mode)
        self.manager = dataset_manager
        self.mode = mode
        self.data = self.manager.data_subset(split=mode).copy()
        self._prepare_data()

        # copy some function from manager for better code readability
        self.rescale_s2 = self.manager.utils.rescale_s2
        self.rescale_s1 = self.manager.utils.rescale_s1
        self.read_tif = self.manager.utils.read_tif
        self.get_cloud_map = self.manager.utils.get_cloud_map
        self.fillnan = self.manager.utils.fillnan

    @staticmethod
    def _check_init_arguments(dataset_manager, mode):

        if not dataset_manager.has_cloud_maps:
            raise ValueError("Provided dataset manager does not contain paths to cloud maps. "
                             "Check whether path to cloud maps directory has been provided to dataset manager.")

        if mode is not None and mode not in ["train", "test", "val"]:
            raise ValueError(f"Incorrect mode provided. "
                             f"Supported modes: None, 'train', 'test', 'val'. Got instead: {mode}")

    def _prepare_data(self):

        self.data["S2_t-1"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(1)
        self.data["S2_t-2"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(2)
        self.data["S2_t-3"] = self.data["S2"].groupby(level=["ROI", "tile", "patch"]).shift(3)

        self.data["S2CLOUDMAP_t-1"] = self.data["S2CLOUDMAP"].groupby(level=["ROI", "tile", "patch"]).shift(1)
        self.data["S2CLOUDMAP_t-2"] = self.data["S2CLOUDMAP"].groupby(level=["ROI", "tile", "patch"]).shift(2)
        self.data["S2CLOUDMAP_t-3"] = self.data["S2CLOUDMAP"].groupby(level=["ROI", "tile", "patch"]).shift(3)

        self.data["S1_t-1"] = self.data["S1"].groupby(level=["ROI", "tile", "patch"]).shift(1)
        self.data["S1_t-2"] = self.data["S1"].groupby(level=["ROI", "tile", "patch"]).shift(2)
        self.data["S1_t-3"] = self.data["S1"].groupby(level=["ROI", "tile", "patch"]).shift(3)

        self.data = self.data.dropna(how="any")

    def sneak_peek(self, idx):
        """
        Allows to take a sneak peek at cloud percentage at specific index.
        This can reduce computation because batch can be discarded without loading the whole sample

        :param idx: same idx as in __getitem__()
        :return: cloud percentage
        """
        index = self.data.index[idx]
        cloud_map = self.get_cloud_map(index=index)
        return cloud_map.mean()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        index = sample.name

        original_s2_image = self.read_tif(sample["S2"])
        original_s1_image = self.fillnan(self.read_tif(sample["S1"]))

        # need this for cloud maps
        original_input_images = [
            self.read_tif(sample["S2_t-1"]),
            self.read_tif(sample["S2_t-2"]),
            self.read_tif(sample["S2_t-3"])
        ]

        original_input_images_s1 = [
            self.fillnan(self.read_tif(sample["S1_t-1"])),
            self.fillnan(self.read_tif(sample["S1_t-2"])),
            self.fillnan(self.read_tif(sample["S1_t-3"]))
        ]

        input_cloud_maps = [
            self.get_cloud_map(cloud_map_path=sample["S2CLOUDMAP_t-1"], s2_image=original_input_images[0]),
            self.get_cloud_map(cloud_map_path=sample["S2CLOUDMAP_t-2"], s2_image=original_input_images[1]),
            self.get_cloud_map(cloud_map_path=sample["S2CLOUDMAP_t-3"], s2_image=original_input_images[2])
        ]

        input_images = [
            self.rescale_s2(image)[self.bands] for image in original_input_images
        ]

        input_images_s1 = [
            self.rescale_s1(image) for image in original_input_images_s1
        ]

        target_image = self.rescale_s2(original_s2_image)[self.bands]
        target_s1_image = self.rescale_s1(original_s1_image)

        cloud_percentage = self.get_cloud_map(
            cloud_map_path=sample["S2CLOUDMAP"],
            s2_image=original_s2_image
        ).mean()

        return {
            "index":                index,
            "original_s2_image":    original_s2_image,
            "target_image":         torch.from_numpy(target_image),
            "target_s1_image":      torch.from_numpy(target_s1_image),
            "inputs":               [
                                        torch.from_numpy(s2_image)
                                        for s2_image in input_images
                                    ],
            "input_cloud_maps":     [
                                        torch.from_numpy(cloud_map)
                                        for cloud_map in input_cloud_maps
                                    ],
            "input_images_s1":      [
                                        torch.from_numpy(s1_image)
                                        for s1_image in input_images_s1
                                    ],
            "cloud_percentage":     cloud_percentage
        }

    @staticmethod
    def collate_fn(list_of_samples):

        result = {
            "index": [],
            "original_s2_image": [],
            "target_image": [],
            "target_s1_image": [],
            "inputs": [[], [], []],
            "input_cloud_maps": [[], [], []],
            "input_images_s1": [[], [], []],
            "cloud_percentage": []
        }
        for sample in list_of_samples:
            result["index"].append(sample["index"])
            result["original_s2_image"].append(sample["original_s2_image"])
            result["target_image"].append(sample["target_image"])
            result["target_s1_image"].append(sample["target_s1_image"])
            result["inputs"][0].append(sample["inputs"][0])
            result["inputs"][1].append(sample["inputs"][1])
            result["inputs"][2].append(sample["inputs"][2])
            result["input_cloud_maps"][0].append(sample["input_cloud_maps"][0])
            result["input_cloud_maps"][1].append(sample["input_cloud_maps"][1])
            result["input_cloud_maps"][2].append(sample["input_cloud_maps"][2])
            result["input_images_s1"][0].append(sample["input_images_s1"][0])
            result["input_images_s1"][1].append(sample["input_images_s1"][1])
            result["input_images_s1"][2].append(sample["input_images_s1"][2])
            result["cloud_percentage"].append(sample["cloud_percentage"])

        result["target_image"] = torch.stack(result["target_image"])
        result["target_s1_image"] = torch.stack(result["target_s1_image"])
        result["inputs"][0] = torch.stack(result["inputs"][0])
        result["inputs"][1] = torch.stack(result["inputs"][1])
        result["inputs"][2] = torch.stack(result["inputs"][2])
        result["input_cloud_maps"][0] = torch.stack(result["input_cloud_maps"][0])
        result["input_cloud_maps"][1] = torch.stack(result["input_cloud_maps"][1])
        result["input_cloud_maps"][2] = torch.stack(result["input_cloud_maps"][2])
        result["input_images_s1"][0] = torch.stack(result["input_images_s1"][0])
        result["input_images_s1"][1] = torch.stack(result["input_images_s1"][1])
        result["input_images_s1"][2] = torch.stack(result["input_images_s1"][2])

        return result


class MinimalTorchDataset(Dataset):

    bands = [3, 2, 1, 7]  # [red, green, blue, NIR]

    def __init__(self, dataset_manager, mode=None):

        self._check_init_arguments(dataset_manager, mode)
        self.manager = dataset_manager
        self.mode = mode
        self.data = self.manager.data_subset(split=mode).copy()

        # copy some function from manager for better code readability
        self.read_tif = self.manager.utils.read_tif
        self.get_cloud_map = self.manager.utils.get_cloud_map
        self.fillnan = self.manager.utils.fillnan

    @staticmethod
    def _check_init_arguments(dataset_manager, mode):

        if not dataset_manager.has_cloud_maps:
            raise ValueError("Provided dataset manager does not contain paths to cloud maps. "
                             "Check whether path to cloud maps directory has been provided to dataset manager.")

        if mode is not None and mode not in ["train", "test", "val"]:
            raise ValueError(f"Incorrect mode provided. "
                             f"Supported modes: None, 'train', 'test', 'val'. Got instead: {mode}")

    def sneak_peek(self, idx):
        """
        Allows to take a sneak peek at cloud percentage at specific index.
        This can reduce computation because batch can be discarded without loading the whole sample

        :param idx: same idx as in __getitem__()
        :return: cloud percentage
        """
        index = self.data.index[idx]
        cloud_map = self.get_cloud_map(index=index)
        return cloud_map.mean()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        sample = self.data.iloc[idx]
        index = sample.name

        original_s2_image = self.read_tif(sample["S2"])
        original_s1_image = self.fillnan(self.read_tif(sample["S1"]))
        cloud_map = self.get_cloud_map(
            cloud_map_path=sample["S2CLOUDMAP"],
            s2_image=original_s2_image
        )
        cloud_percentage = cloud_map.mean()

        return {
            "index":                index,
            "original_s2_image":    original_s2_image,
            "original_s1_image":    original_s1_image,
            "cloud_map":            cloud_map,
            "cloud_percentage":     cloud_percentage
        }

    @staticmethod
    def collate_fn(list_of_samples):

        result = {
            "index": [],
            "original_s2_image": [],
            "original_s1_image": [],
            "cloud_map": [],
            "cloud_percentage": []
        }
        for sample in list_of_samples:
            result["index"].append(sample["index"])
            result["original_s2_image"].append(sample["original_s2_image"])
            result["original_s1_image"].append(sample["original_s1_image"])
            result["cloud_map"].append(sample["cloud_map"])
            result["cloud_percentage"].append(sample["cloud_percentage"])

        result["original_s2_image"] = np.stack(result["original_s2_image"])
        result["original_s1_image"] = np.stack(result["original_s1_image"])
        result["cloud_map"] = np.stack(result["cloud_map"])

        return result


class CTGANTorchIterableDataset(IterableDataset):

    def __init__(
            self,
            dataset_manager=None,
            wrapped_dataset=None,
            mode=None,
            target_cloud_threshold=0.05,
            input_visible_area_threshold=0.5
    ):

        if not dataset_manager and not wrapped_dataset:
            raise ValueError("You need to provide either a dataset manager, or a pytorch dataset")

        if wrapped_dataset:
            self.map_dataset = wrapped_dataset
        else:
            self.map_dataset = CTGANTorchDataset(dataset_manager, mode=mode)

        self.target_cloud_threshold = target_cloud_threshold
        self.input_visible_area_threshold = input_visible_area_threshold
        self.collate_fn = self.map_dataset.collate_fn

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        permutation = np.random.permutation(len(self.map_dataset))

        if worker_info is None:
            iter_start = 0
            iter_end = len(self.map_dataset)
        else:
            per_worker = int(math.ceil(len(self.map_dataset) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = worker_id * per_worker
            iter_end = min(iter_start + per_worker, len(self.map_dataset))

        self.worker_permutation = permutation[iter_start:iter_end]
        self.current_iteration = 0

        return self

    def __next__(self):

        try:
            idx = self.worker_permutation[self.current_iteration]
        except IndexError:
            raise StopIteration

        self.current_iteration += 1

        # dismiss samples where target cloud cover is too high
        target_cloud_percentage = self.map_dataset.sneak_peek(idx)
        if target_cloud_percentage > self.target_cloud_threshold:
            return self.__next__()

        sample = self.map_dataset[idx]

        # dismiss samples where too much area was always cloudy in all three images, hence too little visible area
        if "input_cloud_maps" in sample:
            multiplied_clouds = sample["input_cloud_maps"][0] * sample["input_cloud_maps"][1] * sample["input_cloud_maps"][2]
            visible_area = (multiplied_clouds < 0.5).float().mean()
            if visible_area < self.input_visible_area_threshold:
                return self.__next__()

        return sample

    def __len__(self):
        return len(self.map_dataset)

    def __add__(self, other):
        raise NotImplementedError("Adding datasets together is not implemented yet")
