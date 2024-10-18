from .imagefile import ImageFile
from scipy.interpolate import NearestNDInterpolator
from os import makedirs
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from os.path import join, isfile
from tqdm import tqdm


class Modification(ABC):

    def __init__(self, dataset_manager):
        self.dataset_manager = dataset_manager
        self.utils = self.dataset_manager.utils
        self.mod_dir = self.dataset_manager.root_dir + "_MODS"

    @abstractmethod
    def apply(self):
        raise NotImplementedError


class ZeroPixelsS2(Modification):

    def __init__(self, dataset_manager):

        super().__init__(dataset_manager)

        self.mod_dir = join(self.mod_dir, type(self).__name__)
        self.config_path = join(self.dataset_manager.project_directory, "config", f"MOD_{type(self).__name__}.csv")

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
            self.modification = pd.read_csv(
                self.config_path,
                index_col=["ROI", "tile", "patch", "timestep"]
            )
            self.recalculate = False
        except FileNotFoundError:
            self.modification = pd.DataFrame(
                pd.NA,
                index=self.zero_patches.index,
                columns=["S2", "S2CLOUDMAP"]
            )
            self.recalculate = True

    def apply(self, verbose=False):

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

    def _save(self):
        self.modification.to_csv(self.config_path)
