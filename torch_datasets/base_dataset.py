import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import pandas as pd
from copy import copy
from typing import List


class BaseDataset(Dataset, ABC):

    def __init__(
            self,
            dataset_manager
    ):
        super().__init__()
        self.manager = dataset_manager
        self.data = self.initialize_data()

        # Add t_shift level to columns
        if "t_shift" not in self.data.columns.names:
            self.data.columns = pd.MultiIndex.from_product(
                iterables=[[0], self.data.columns],
                names=["t_shift", "modality"]
            )
            self.data[(0, "index_tuple")] = self.data.index

    @abstractmethod
    def initialize_data(self):
        raise NotImplementedError

    def shift(self, t_shift: int, inplace=False):

        if not inplace:
            return copy(self).shift(t_shift=t_shift, inplace=True)

        # shift data
        shifted_data = self.data.groupby(level=["ROI", "tile", "patch"]).shift(-t_shift)

        # change values in t_shift column level
        shifted_data.columns = shifted_data.columns.set_levels(
            shifted_data.columns.levels[0] + t_shift,
            level="t_shift"
        )
        # We don't want to drop NaNs now, because there can be other NaNs in other rows which we might want to retain
        # self.data = shifted_data.dropna(axis=0)

        return self

    def dropna(self):
        self.data = self.data.dropna(axis=0, how="any")

    def __add__(self, other):
        return MergedDataset([self, other])


class MergedDataset(BaseDataset):

    def __init__(
            self,
            datasets: List[BaseDataset]
    ):
        managers = set([dataset.manager for dataset in datasets])
        if not len(managers) == 1:
            raise ValueError("Provided pytorch datasets have different dataset managers!")

        self.children = datasets

        super().__init__(dataset_manager=managers.pop())

    def initialize_data(self):
        return pd.concat([child.data for child in self.children], join="outer", axis=1)


class S1(BaseDataset):

    def initialize_data(self):
        return self.manager.data[["S1"]]


class S2(BaseDataset):

    def initialize_data(self):
        return self.manager.data[["S2"]]


class S2CloudMap(BaseDataset):

    def initialize_data(self):
        return self.manager.data[["S2CLOUDMAP"]]