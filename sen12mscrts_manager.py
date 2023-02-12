import numpy as np
import pandas as pd
import os
from os import makedirs
from os.path import join, isfile, isdir, dirname, abspath
import warnings
import yaml
from image import ImageFile
from tqdm import tqdm
import rasterio
from rasterio import RasterioIOError
from s2cloudless import S2PixelCloudDetector
from scipy.ndimage import gaussian_filter


class Sen12mscrtsDatasetManager:
    """
    File manager which finds SEN12MS-CR-TS dataset files, parses some metadata, and stores it in the registry.
    Registry is stored as a MultiIndex pd.DataFrame under the .registry variable.
    """

    # load class config file
    project_directory = os.path.abspath(os.path.dirname(__file__))
    with open(join(project_directory, "sen12mscrts.yaml"), 'r') as file:
        config = yaml.safe_load(file)

    # load definition of data subsets (regions and test/val splits)
    subsets = pd.read_csv(join(project_directory, "subsets.csv"), index_col=["ROI", "tile"])

    def __init__(
            self,
            root_dir,
            cloud_maps_dir=None,
            cloud_percentage_csv=None,
            cloud_percentage_buffer=None
    ):
        if not isdir(root_dir):
            raise ValueError(f"Provided root directory does not exist: {root_dir}")
        self.root_dir = root_dir
        self.cloud_maps_dir = cloud_maps_dir
        self.cloud_percentage_csv = cloud_percentage_csv

        self._files = {}
        self._data = None

        self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        self.utils = ImageUtils(manager=self)

    @property
    def data(self):
        """ Getter method to prevent outer classes from editing the registry """
        return self._data

    @property
    def has_cloud_maps(self):
        return self._data is not None and "S2CLOUDMAP" in self._data

    def load_dataset(self):
        self.get_paths_to_files()
        self.build_dataframe()

    def get_paths_to_files(self):
        """
        Creates a pd.DataFrame, finds all dataset files in root directory, and adds them to the dataframe
        :return:
        """

        with tqdm(total=sum([len(files) for r, d, files in os.walk(self.root_dir)])) as pbar:

            for current_path, directories, filenames in os.walk(self.root_dir):

                for filename in filenames:

                    pbar.update(1)

                    # skip everything except for .tif images
                    if not filename.endswith(".tif"):
                        continue

                    image_file = ImageFile(manager=self, directory=current_path, filename=filename)

                    self._files[image_file.index] = image_file.filepath

                    # add path to cloud map if applicable
                    # if no .tif image is present at that path, it will be used as target path to generate cloud map
                    if self.cloud_maps_dir and image_file.optical:
                        self._files[image_file.cloud_map_index] = image_file.path_to_cloud_map

    def build_dataframe(self):

        # put filenames into a pd.Series
        self._data = pd.Series(
            index=pd.MultiIndex.from_tuples(self._files.keys(), names=self.config["dataset_index"]),
            data=self._files.values(),
            dtype="object"
        )

        # sort
        self._data = self._data.sort_index()

        # put modality into columns (creates a pd.DataFrame
        self._data = self._data.unstack("modality")

    def data_subset(self, split=None, only_resampled=True):

        if split is not None:
            subset = self.subsets[self.subsets["split"] == split]
        else:
            subset = self.subsets

        if only_resampled:
            subset = subset[subset["resampled"] == True]

        index_tuples = [index_tuple for index_tuple in subset.index]

        iloc_indices = []
        for index_tuple in index_tuples:
            try:
                iloc_indices.append(self.data.index.get_locs(index_tuple))
            except KeyError:
                continue
        indices = np.concatenate(iloc_indices) if iloc_indices else []

        return self.data.iloc[indices]

# ####################### ImageUtils ###########################


class ImageUtils:

    project_directory = os.path.abspath(os.path.dirname(__file__))
    min_max = pd.read_csv(join(project_directory, "98_percentile_min_max.csv"), index_col="band")

    def __init__(self, manager):
        self.manager = manager

    def get_cloud_map(self, cloud_map_path=None, s2_image=None, index=None, threshold=0.5):

        # handle index and cloud_map_path
        if index and not cloud_map_path:
            try:
                cloud_map_path = self.manager.data.loc[index]["S2CLOUDMAP"]
            except KeyError:
                raise ValueError(f"Can not find path to cloud mask in the dataset using provided index: {index}")

        if cloud_map_path and not index:
            index = ImageFile(manager=self.manager, filepath=cloud_map_path).short_index
            if index not in self.manager.data.index:
                raise ValueError(f"Can not find provided cloud map path in the dataset: {cloud_map_path}")

        # try simply loading cloud map from disk
        if cloud_map_path:
            try:
                return self.read_tif(filepath=cloud_map_path)
            except RasterioIOError:
                pass

        # try to retrieve path to s2 image
        path_to_s2 = None
        if index:
            try:
                path_to_s2 = self.manager.data.loc[index]["S2"]
            except KeyError:
                pass

        # load s2 image if not loaded yet
        if s2_image is None:
            if path_to_s2 is None:
                raise ValueError("Cloud not find the corresponding Sentinel-2 image in the dataset")
            try:
                s2_image = self.read_tif(path_to_s2)
            except RasterioIOError:
                raise ValueError("Could not find the corresponding Sentinel-2 image under the path from the dataset")

        # at this point it is guaranteed that s2_image has been loaded
        # calculate cloud map
        s2_image = self.prepare_for_cloud_detector(s2_image)
        cloud_map = self.manager.cloud_detector.get_cloud_probability_maps(s2_image)
        cloud_map = cloud_map[np.newaxis, ...]
        cloud_map[cloud_map < threshold] = 0
        cloud_map = gaussian_filter(cloud_map, sigma=2).astype(np.float32)

        # try to save cloud mask
        # we can only save to .tif file if we have an example Sentinel-2 .tif file to copy its rasterio profile
        if cloud_map_path and path_to_s2:
            rasterio_profile = self.get_rasterio_profile(path_to_s2)

            self.save_tif(
                filepath=cloud_map_path,
                image=cloud_map,
                rasterio_profile=rasterio_profile,
                dtype=rasterio.float32
            )

        # try to store cloud percentage
        # if index:
        #    self.register_cloud_percentage(index, cloud_percentage=cloud_map.mean())

        return cloud_map

    @staticmethod
    def read_tif(filepath):
        reader = rasterio.open(filepath)
        return reader.read()

    @staticmethod
    def get_rasterio_profile(filepath):
        reader = rasterio.open(filepath)
        return reader.profile

    @staticmethod
    def save_tif(
        filepath,
        image,
        rasterio_profile,
        dtype=None
    ):

        if image.ndim == 2:
            image = image[np.newaxis]

        if not image.ndim == 3:
            raise ValueError(f"Can only save images of shape [height, width] or [band, height, width]. "
                             f"Got instead: {image.shape}")

        if not rasterio_profile["height"] == image.shape[1] or not rasterio_profile["width"] == image.shape[2]:
            raise ValueError(f"Height and width of rasterio profile "
                             f"({rasterio_profile['height']}, {rasterio_profile['width']}) "
                             f"and provided array ({image.shape[1]}, {image.shape[2]}) do not match")

        if dtype:
            rasterio_profile["dtype"] = dtype

        rasterio_profile["count"] = image.shape[0]

        filepath = abspath(filepath)
        makedirs(dirname(filepath), exist_ok=True)

        with rasterio.open(filepath, 'w', **rasterio_profile) as dst:
            dst.write(image)

    @staticmethod
    def bands_last(s2_image):
        return np.transpose(s2_image, (1, 2, 0))

    @classmethod
    def prepare_for_cloud_detector(cls, s2_image):
        s2_image = cls.bands_last(s2_image)
        return s2_image.clip(0, 10000) / 10000

    @classmethod
    def rescale_s2(cls, s2_image):

        if not s2_image.shape[0] == 13 or not s2_image.ndim == 3:
            raise ValueError(f"Only accept images of shape [band, height, width] with 13 bands. "
                             f"Got instead: {s2_image.shape}")

        bands_min = cls.min_max["min"].values[:, np.newaxis, np.newaxis]
        bands_max = cls.min_max["max"].values[:, np.newaxis, np.newaxis]

        rescaled_image = s2_image.clip(min=bands_min, max=bands_max)
        rescaled_image = rescaled_image - bands_min
        rescaled_image = rescaled_image / (bands_max - bands_min)
        rescaled_image = rescaled_image - 0.5
        rescaled_image = rescaled_image * 2

        return rescaled_image

    @classmethod
    def rescale_s2_back(cls, rescaled_s2_image):

        bands_min = cls.min_max["min"].values[:, np.newaxis, np.newaxis]
        bands_max = cls.min_max["max"].values[:, np.newaxis, np.newaxis]

        back_rescaled_image = rescaled_s2_image / 2
        back_rescaled_image = back_rescaled_image + 0.5
        back_rescaled_image = back_rescaled_image * (bands_max - bands_min)
        back_rescaled_image = back_rescaled_image + bands_min

        return np.round(back_rescaled_image).astype('uint16')