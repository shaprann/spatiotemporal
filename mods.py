from .imagefile import ImageFile
from scipy.interpolate import NearestNDInterpolator
from os import makedirs
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from os.path import join, isfile
from tqdm import tqdm


class Modification(ABC):

    requirements = tuple()

    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
        self.utils = self.dataset_manager.utils
        self.mod_dir = self.dataset_manager.root_dir + "_MODS"

    def check_requirements(self):
        for requirement in self.requirements:
            if requirement not in self.dataset_manager._modifications:
                raise ValueError(f"Requirement not fulfilled! First apply modification '{requirement}' to dataset")

    @property
    @abstractmethod
    def modification(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def config_path(self):
        raise NotImplementedError

    @abstractmethod
    def _apply(self, verbose=None):
        raise NotImplementedError

    def apply(self, verbose=None):
        self._apply(verbose=verbose)
        self.dataset_manager._modifications.append(type(self).__name__)

    def apply_modification(self, verbose=None):
        """ Alias for apply() """
        self.apply(verbose=verbose)

    def _save(self):
        self.modification.to_csv(self.config_path)


class ZeroPixelsS2(Modification):

    def __init__(self, dataset_manager):

        super().__init__(dataset_manager)

        self.mod_dir = join(self.mod_dir, type(self).__name__)
        self._config_path = join(self.dataset_manager.project_directory, "config", f"MOD_{type(self).__name__}.csv")

        self.zero_patches = pd.read_csv(
            join(self.dataset_manager.project_directory, "stats", "S2_patches_with_zeros.csv"),
            index_col=["ROI", "tile", "patch", "timestep"]
        )
        self.zero_pixels = pd.read_csv(
            join(self.dataset_manager.project_directory, "stats", "S2_zero_pixels_count.csv"),
            index_col=["ROI", "tile", "patch", "timestep"]
        )
        self.patches_to_interpolate = self.zero_patches[~self.zero_patches["Empty"] & ~self.zero_patches["Border"]].index

        try:
            self._modification = pd.read_csv(
                self.config_path,
                index_col=["ROI", "tile", "patch", "timestep"]
            )
            self.recalculate = False
        except FileNotFoundError:
            self._modification = pd.DataFrame(
                pd.NA,
                index=self.zero_patches.index,
                columns=["S2", "S2CLOUDMAP"]
            )
            self.recalculate = True

    @property
    def modification(self):
        return self._modification

    @property
    def config_path(self):
        return self._config_path

    def _apply(self, verbose=False):

        if verbose:
            print("Fixing zero pixels in S2 images...")

        if self.recalculate:
            self._interpolate_patches(verbose=verbose)
            self._remove_empty_tiles()
            self._save()

        self.dataset_manager._data.loc[self.modification.index, self.modification.columns] = self.modification

    def _interpolate_patches(self, verbose):

        patch_indices = self.patches_to_interpolate
        if verbose:
            patch_indices = tqdm(patch_indices, desc="Interpolating patches ")

        for patch_index in patch_indices:

            path_to_s2 = self.dataset_manager.data.loc[patch_index, "S2"]
            path_to_cloudmap = self.dataset_manager.data.loc[patch_index, "S2CLOUDMAP"]
            s2_outfile = ImageFile(filepath=path_to_s2).set(root_dir=self.mod_dir)
            cloudmap_outfile = ImageFile(filepath=path_to_cloudmap).set(root_dir=self.mod_dir)
            s2_image = None

            # add paths to modification
            self.modification.loc[patch_index, "S2"] = s2_outfile.filepath
            self.modification.loc[patch_index, "S2CLOUDMAP"] = cloudmap_outfile.filepath

            # skip interpolation if interpolated file already exists
            if not isfile(s2_outfile.filepath):

                s2_image = self.utils.read_tif(path_to_s2)
                bands_to_interpolate = (self.zero_pixels.loc[patch_index] != 0).values
                bands_to_interpolate = np.flatnonzero(bands_to_interpolate)

                self._interpolate_zeros_inplace(s2_image, bands_to_interpolate)

                makedirs(s2_outfile.directory, exist_ok=True)

                self.utils.save_tif(
                    filepath=s2_outfile.filepath,
                    image=s2_image,
                    rasterio_profile=self.utils.get_rasterio_profile(path_to_s2),
                    dtype=s2_image.dtype
                )

            if not isfile(cloudmap_outfile.filepath):

                if s2_image is None:
                    s2_image = self.utils.read_tif(path_to_s2)

                makedirs(cloudmap_outfile.directory, exist_ok=True)

                cloud_map = self.dataset_manager.cloud_detector.get_cloud_probability_maps(s2_image)

                self.utils.save_tif(
                    filepath=cloudmap_outfile.filepath,
                    image=cloud_map,
                    rasterio_profile=self.utils.get_rasterio_profile(path_to_cloudmap),
                    dtype=self.dataset_manager.cloud_detector.dtype
                )

    @staticmethod
    def _interpolate_zeros_inplace(image, bands):

        for band in bands:
            band_image = image[band]
            zero_mask = band_image == 0
            interpolator = NearestNDInterpolator(
                x=np.argwhere(~zero_mask),
                y=band_image[~zero_mask]
            )
            image[band, zero_mask] = interpolator(np.argwhere(zero_mask))

    def _remove_empty_tiles(self):

        empty_patches = self.zero_patches[self.zero_patches["Empty"]].index
        self.modification.loc[empty_patches, ["S2", "S2CLOUDMAP"]] = pd.NA

        border_patches = self.zero_patches[self.zero_patches["Border"]].index
        self.modification.loc[border_patches, ["S2", "S2CLOUDMAP"]] = pd.NA


class CategoricalCloudMaps(Modification):

    requirements = (
        ZeroPixelsS2.__name__,
    )

    def __init__(self, dataset_manager):

        super().__init__(dataset_manager)
        self.check_requirements()
        self.mod_dir = join(self.mod_dir, type(self).__name__)
        self._config_path = join(self.dataset_manager.project_directory, "config", f"MOD_{type(self).__name__}.csv")
        try:
            self._modification = pd.read_csv(
                self.config_path,
                index_col=["ROI", "tile", "patch", "timestep"]
            )
        except FileNotFoundError:
            self._modification = None

    @property
    def modification(self):
        return self._modification

    @property
    def config_path(self):
        return self._config_path

    def _apply(self, verbose=False):

        if verbose:
            print("Adding categorical cloud maps...")

        if self.modification is None:

            if verbose:
                tqdm.pandas(desc="Adding categorical cloud maps... ")
                self._modification = self.dataset_manager.data["S2CLOUDMAP"].progress_apply(self._transform_cloud_path)
            else:
                self._modification = self.dataset_manager.data["S2CLOUDMAP"].apply(self._transform_cloud_path)

            self._save()

        self.dataset_manager._data["S2CLOUDMAP"] = self.modification

    def _transform_cloud_path(self, filepath):

        if filepath is np.nan:
            return None
        outfile = ImageFile(filepath)
        outfile = outfile.set(root_dir=self.mod_dir)
        filepath = outfile.filepath
        if isfile(filepath):
            return filepath
        else:
            return None


class CloudfreeArea(Modification):

    requirements = (
        ZeroPixelsS2.__name__,
        CategoricalCloudMaps.__name__,
    )

    def __init__(self, dataset_manager):

        super().__init__(dataset_manager)
        self.check_requirements()
        self.mod_dir = join(self.mod_dir, type(self).__name__)
        self.threshold = int(dataset_manager.cloud_probability_threshold * 100)
        self.cloud_probability_histogram = pd.read_csv(
            join(self.dataset_manager.project_directory, "stats", "cloud_probability_histogram_int.csv"),
            index_col=["ROI", "tile", "patch", "timestep"]
        )
        self._modification = pd.Series(
            pd.NA,
            index=self.dataset_manager.data.index,
            name="CLOUDFREEAREA"
        )
        self.cloud_probability_cumulative = np.cumsum(self.cloud_probability_histogram.values, axis=-1)

    @property
    def modification(self):
        return self._modification

    @property
    def config_path(self):
        raise NotImplementedError

    def _apply(self, verbose=False):

        if verbose:
            print("Adding cloudfree area percentages...")

        self._modification.loc[self.cloud_probability_histogram.index] = (
                self.cloud_probability_cumulative[:, self.threshold] / self.cloud_probability_cumulative[:, -1]
        )
        self.dataset_manager._data["CLOUDFREEAREA"] = self._modification
