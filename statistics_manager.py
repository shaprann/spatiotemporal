from abc import ABC, abstractmethod
from contextlib import contextmanager
from os.path import join, isfile
import pandas as pd
import rasterio


class StatisticsManager:

    def __init__(self, manager, **kwargs):

        self.manager = manager
        self.statistics_directory = join(manager.project_directory, "stats")

        self.with_overwrite = False
        self.with_compute = False
        self.with_progressbar = None

        self._available_statistics = self._initialize_statistics(**kwargs)

        # TODO: how to do it best?
        # self._crs_transforms_csv = join(self.manager.project_directory, "stats", "crs_transforms.csv")

    @property
    def available_statistics(self):
        return list(self._available_statistics.keys())

    @contextmanager
    def force_overwrite(self):
        self.with_overwrite = True
        try:
            yield
        finally:
            self.with_overwrite = False

    @contextmanager
    def force_compute(self):
        self.with_compute = True
        try:
            yield
        finally:
            self.with_compute = False

    @contextmanager
    def progressbar(self, progressbar_object):
        self.with_progressbar = progressbar_object
        try:
            yield
        finally:
            self.with_progressbar = None

    def _initialize_statistics(self, **kwargs):
        return {
            statistic.name: statistic(statistics_manager=self, **kwargs)
            for statistic in Statistic.__subclasses__()
        }

    def get_statistic_object(self, name):
        return self._available_statistics[name]

    def __getattr__(self, name):
        try:
            return self._available_statistics[name].get()
        except KeyError:
            return self.__getattribute__(name)  # this raises a descriptive error message

    def __repr__(self):
        return f"<{self.__class__.__name__} object>\nAvailable statistics: {self.available_statistics}"


class Statistic(ABC):

    def __init__(
            self,
            statistics_manager,
            **kwargs
    ):
        self.statistics_manager = statistics_manager
        self.manager = self.statistics_manager.manager

        self.path_to_csv = self._get_path_to_csv(**kwargs)
        self._requirements_loaded = None

    def _get_path_to_csv(self, **kwargs):
        try:
            return kwargs[self.name + "_csv"]
        except KeyError:
            return join(self.statistics_manager.statistics_directory, self.name + ".csv")

    def get(self):
        return self.compute() if self.statistics_manager.with_compute else self.read()

    def read(self):
        try:
            return self._read()
        except FileNotFoundError:
            raise FileNotFoundError(f"Can not read {self.name}: CSV file for {self.name} does not exist yet.\n"
                                    f"Use context manager 'force_compute()' to compute {self.name} instead")

    def compute(self):

        try:
            self._load_requirements()
        except FileNotFoundError as err:
            raise FileNotFoundError(f"Can not compute {self.name} because some requirements are missing.") from err

        print(f"Compute {self.name} for the whole dataset...")

        if self.statistics_manager.with_overwrite and isfile(self.path_to_csv):
            print("Warning: the existing CSV file will be overwritten!")

        result = self._compute()

        if self.statistics_manager.with_overwrite or not isfile(self.path_to_csv):
            result.to_csv(self.path_to_csv)

        return result

    def _load_requirements(self):
        if self.requires is None:
            self._requirements_loaded = {}
            return
        requirement_names = self.requires if isinstance(self.requires, list) else [self.requires]
        self._requirements_loaded = {
            requirement_name: self.statistics_manager.get_statistic_object(name=requirement_name).read()
            for requirement_name in requirement_names
        }

    @property
    @abstractmethod
    def name(self):
        pass

    @property
    @abstractmethod
    def requires(self):
        pass

    @abstractmethod
    def _read(self):
        pass

    @abstractmethod
    def _compute(self):
        pass


class CrsTransform(Statistic):

    name = "crs_transform"
    requires = None

    def _read(self):
        return pd.read_csv(
            self.path_to_csv,
            header=[0, 1],
            index_col=[0, 1, 2, 3],
            skipinitialspace=True,
        )

    def _compute(self):

        crs_transform_df = pd.DataFrame(
            data=None,
            columns=pd.MultiIndex.from_product(
                iterables=[
                    ["S1", "S2"],
                    ["crs"] + list(range(6))
                ],
                names=["modality", "info"]
            ),
            index=self.manager.data.index
        )

        patch_indices = self.manager.data.index.droplevel("timestep").unique()

        if self.statistics_manager.with_progressbar is not None:
            patch_indices = self.statistics_manager.with_progressbar(patch_indices)

        for patch_index in patch_indices:

            patch_timesteps = self.manager.data[
                self.manager.data.index.droplevel("timestep").get_loc(patch_index)
            ]

            for index, filepaths in patch_timesteps.iterrows():

                for modality in ["S1", "S2"]:
                    with rasterio.open(filepaths[modality]) as reader:
                        profile = reader.profile
                        crs_transform_df.loc[index, (modality, "crs")] = str(profile["crs"])
                        crs_transform_df.loc[index, (modality, list(range(6)))] = profile["transform"][:6]

        return crs_transform_df


class S2Resampled(Statistic):

    name = "S2_resampled"
    requires = "crs_transform"

    def _read(self):
        return pd.read_csv(self.path_to_csv, index_col=["ROI", "tile"])

    def _compute(self):
        crs_transform_df = self._requirements_loaded["crs_transform"]
        s2_patches_unique_crs_transform = crs_transform_df["S2"].groupby(level=["ROI", "tile", "patch"]).nunique()
        s2_patches_resampled = (s2_patches_unique_crs_transform == 1).all(axis=1).to_frame(name="S2_resampled")
        s2_tiles_resampled = s2_patches_resampled.groupby(level=["ROI", "tile"]).all()
        return s2_tiles_resampled


class S1Resampled(Statistic):

    name = "S1_resampled"
    requires = "crs_transform"

    def _read(self):
        return pd.read_csv(self.path_to_csv, index_col=["ROI", "tile"])

    def _compute(self):
        crs_transform_df = self._requirements_loaded["crs_transform"]
        s1_patches_unique_crs_transform = crs_transform_df["S1"].groupby(level=["ROI", "tile", "patch"]).nunique()
        s1_patches_resampled = (s1_patches_unique_crs_transform == 1).all(axis=1).to_frame(name="S1_resampled")
        s1_tiles_resampled = s1_patches_resampled.groupby(level=["ROI", "tile"]).all()
        return s1_tiles_resampled

