from .imagefile import ImageFile

import yaml
import rasterio
import numpy as np
import pandas as pd
from os import walk, makedirs
from os.path import join, isdir, dirname, abspath, isfile
from tqdm import tqdm
from scipy.ndimage import convolve
from skimage.morphology import dilation, disk
from rasterio import RasterioIOError
from s2cloudless import S2PixelCloudDetector
from s2cloudless.utils import MODEL_BAND_IDS


class DatasetManager:
    """
    File manager which finds SEN12MS-CR-TS dataset files, parses some metadata, and stores it in the registry.
    Registry is stored as a MultiIndex pd.DataFrame under the ._data variable.
    """

    # get current project directory
    project_directory = abspath(dirname(__file__))

    # load dataset config file
    with open(join(project_directory, "config/sen12mscrts.yaml"), 'r') as file:
        config = yaml.safe_load(file)

    # load definition of data subsets (regions and test/val splits)
    subsets = pd.read_csv(join(project_directory, "config/subsets.csv"), index_col=["ROI", "tile"])

    def __init__(
            self,
            root_dir,
            cloud_maps_dir=None,
            cloud_probability_threshold=None
    ):

        if not isdir(root_dir):
            raise ValueError(f"Provided root directory does not exist: {root_dir}")
        self.root_dir = root_dir
        self.cloud_maps_dir = cloud_maps_dir

        self.utils = ImageUtils(manager=self)

        self._files = {}
        self._data = None

        self.cloud_detector = S2PixelCloudDetectorWrapper(
            threshold=cloud_probability_threshold,
            all_bands=True,
            average_over=4,
            dilation_size=2
        )

    @property
    def data(self):
        """ Getter method to prevent outer classes from editing the data """
        if self._data is None:
            raise ValueError("Dataset contains no data yet. Try running 'load_dataset()' method first.")
        return self._data.copy(deep=True)

    @property
    def has_cloud_maps(self):
        return self._data is not None and "S2CLOUDMAP" in self._data

    @property
    def cloud_probability_threshold(self):
        if self.cloud_detector.threshold is None:
            raise ValueError("Cloud probability threshold has not been set yet")
        return self.cloud_detector.threshold

    @cloud_probability_threshold.setter
    def cloud_probability_threshold(self, threshold):
        self.cloud_detector.threshold = threshold

    def load_dataset(self):
        self.get_paths_to_files()
        self.build_dataframe()

    def load_from_file(self, filepath=None):
        if filepath is None:
            filepath = join(self.project_directory, "config", "dataset_manager.csv")
        self._data = pd.read_csv(
            filepath,
            index_col=[idx for idx in self.config["dataset_index"] if not idx == "modality"]
        )

    def save_to_file(self, filepath=None):
        if filepath is None:
            filepath = join(self.project_directory, "config", "dataset_manager.csv")
        self.data.to_csv(filepath)

    def get_paths_to_files(self):
        """
        Creates a pd.DataFrame, finds all dataset files in root directory, and adds them to the dataframe
        :return:
        """

        with tqdm(total=sum([len(files) for r, d, files in walk(self.root_dir)])) as pbar:

            for current_path, directories, filenames in walk(self.root_dir):

                for filename in filenames:

                    pbar.update(1)

                    # skip everything except for .tif images
                    if not filename.endswith(".tif"):
                        continue

                    image_file = ImageFile(manager=self, directory=current_path, filename=filename)

                    self._files[image_file.index] = image_file.filepath

                    # add path to cloud map if applicable
                    # if no .tif image is present at that path, it will be used as target path to generate cloud map
                    # TODO: this will not work if cloud maps are in the same folder as other images.
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
        return self.get_subset(self.data, split=split, only_resampled=only_resampled)


    def get_subset(self, dataframe, split=None, only_resampled=True):

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

        return dataframe.iloc[indices].copy(deep=True)

# ####################### ImageUtils ###########################


class ImageUtils:

    project_directory = abspath(dirname(__file__))
    min_max_s2 = pd.read_csv(join(project_directory, "stats/S2_99_percentile_min_max.csv"), index_col="band")
    min_max_s1 = pd.read_csv(join(project_directory, "stats/S1_99_percentile_min_max.csv"), index_col="band")

    def __init__(self, manager):
        self.manager = manager

    def get_cloud_map(
            self,
            cloud_map_path=None,
            s2_image=None,
            index=None,
            try_load=True,
            overwrite=False
    ):
        """
        Cloud maps are computed and stored in the same format as used by Google Earth Engine:
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY#description
        Namely, probabilities are rescaled to [0..100], discretized to uint8, and value 255 is used as NOVALUE mask
        """

        # if index is provided, ignore cloud_map_path argument, get path from the dataset instead, and overwrite it
        if index is not None:
            try:
                cloud_map_path = self.manager.data.loc[index]["S2CLOUDMAP"]
            except KeyError:
                raise ValueError(f"Can not find path to cloud mask in the dataset using provided index: {index}")

        # if only cloud_map_path is provided, retrieve dataset index for the cloud map path
        if cloud_map_path is not None and index is None:
            index = ImageFile(manager=self.manager, filepath=cloud_map_path).short_index
            if index not in self.manager.data.index:
                raise ValueError(f"Can not find provided cloud map path in the dataset: {cloud_map_path}")

        # now we either have both index and cloud_map_path, or we have neither
        # if we have the cloud_map_path, try simply loading cloud map from the drive
        if try_load and cloud_map_path is not None:
            try:
                return self.read_tif(filepath=cloud_map_path)
            except RasterioIOError:
                pass

        # If we failed to load cloud map from drive, calculate it from S2 image and save to drive if possible

        # First, try to retrieve path to s2 image
        path_to_s2 = None
        if index is not None:
            try:
                path_to_s2 = self.manager.data.loc[index]["S2"]
            except KeyError:
                pass

        # Then, if S2 image was not provided as argument already, try to load it from drive
        if s2_image is None:
            if path_to_s2 is None:
                raise ValueError("Cloud not find the corresponding Sentinel-2 image in the dataset")

            try:
                s2_image = self.read_tif(path_to_s2)
            except RasterioIOError:
                raise ValueError("Could not find the corresponding Sentinel-2 image under the path from the dataset")

        # At this point it is guaranteed that either s2_image has been loaded, or an error was raised.

        # we can finally compute the cloud map from s2_image
        cloud_map = self.manager.cloud_detector.get_cloud_probability_maps(s2_image)

        # Try to save the newly computed cloud map to drive
        # We can only save to .tif if we have an example Sentinel-2 .tif file to copy its rasterio profile
        if overwrite and cloud_map_path is not None and path_to_s2 is not None:
            rasterio_profile = self.get_rasterio_profile(path_to_s2)
            self.save_tif(
                filepath=cloud_map_path,
                image=cloud_map,
                rasterio_profile=rasterio_profile,
                dtype=rasterio.uint8
            )

        return cloud_map

    def get_cloud_mask(
            self,
            threshold=None,
            average_over=None,
            dilation_size=None,
            cloud_map=None,
            cloud_map_path=None,
            s2_image=None,
            index=None,
            try_load=True,
            overwrite=False,
    ):

        if cloud_map is None:
            try:
                cloud_map = self.get_cloud_map(
                    cloud_map_path=cloud_map_path,
                    s2_image=s2_image,
                    index=index,
                    try_load=try_load,
                    overwrite=overwrite
                )
            except ValueError as err:
                raise ValueError("Can not calculate cloud mask, "
                                 "because reading or calculating cloud probability maps failed") from err

        return self.manager.cloud_detector.get_mask_from_prob(
            cloud_map,
            threshold=threshold,
            average_over=average_over,
            dilation_size=dilation_size
        )

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
    def threshold(cls, image, threshold_value):
        return image > threshold_value

    @classmethod
    def prepare_for_cloud_detector(cls, s2_image):
        s2_image = cls.bands_last(s2_image)
        return s2_image.clip(0, 10000) / 10000

    @staticmethod
    def rescale(image, bands_min, bands_max, clip=False):

        if not image.ndim == 3:
            raise ValueError(f"Only accept images of shape [band, height, width]. "
                             f"Got instead: {image.shape}")
        if not image.shape[0] == bands_min.shape[0] == bands_max.shape[0]:
            raise ValueError(f"Number of image bands must be equal to number of values of bands_min and bands_max."
                             f"Got instead: {image.shape[0]}, {bands_min.shape[0]}, {bands_max.shape[0]}")

        bands_min = np.array(bands_min)[:, np.newaxis, np.newaxis]
        bands_max = np.array(bands_max)[:, np.newaxis, np.newaxis]

        if clip:
            image = image.clip(min=bands_min, max=bands_max)
        image = image - bands_min
        image = image / (bands_max - bands_min)

        return image

    @classmethod
    def rescale_s2(cls, s2_image, clip=False):

        if not s2_image.shape[0] == 13:
            raise ValueError(f"Only accept images of shape [band, height, width] with 13 bands. "
                             f"Got instead: {s2_image.shape}")
        return cls.rescale(
            image=s2_image,
            bands_min=cls.min_max_s2["min"],
            bands_max=cls.min_max_s2["max"],
            clip=clip
        )

    @classmethod
    def rescale_s1(cls, s1_image, clip=False):

        if not s1_image.shape[0] == 2:
            raise ValueError(f"Only accept images of shape [band, height, width] with 13 bands. "
                             f"Got instead: {s1_image.shape}")
        return cls.rescale(
            image=s1_image,
            bands_min=cls.min_max_s1["min"],
            bands_max=cls.min_max_s1["max"],
            clip=clip
        )

    @staticmethod
    def rescale_back(image, bands_min, bands_max, bands=None):

        if not image.ndim == 3:
            raise ValueError("Image must have 3 dimensions")

        bands_min = np.array(bands_min)[:, np.newaxis, np.newaxis]
        bands_max = np.array(bands_max)[:, np.newaxis, np.newaxis]

        if bands is not None:
            bands_min = bands_min[bands]
            bands_max = bands_max[bands]

        image = image * (bands_max - bands_min)
        image = image + bands_min

        return image

    @classmethod
    def rescale_s2_back(cls, rescaled_s2_image, bands=None):

        result = cls.rescale_back(
            image=rescaled_s2_image,
            bands_min=cls.min_max_s2["min"],
            bands_max=cls.min_max_s2["max"],
            bands=bands
        )
        return np.round(result).astype('uint16')

    @classmethod
    def rescale_s1_back(cls, rescaled_s1_image, bands=None):

        return cls.rescale_back(
            image=rescaled_s1_image,
            bands_min=cls.min_max_s1["min"],
            bands_max=cls.min_max_s1["max"],
            bands=bands
        )

    @staticmethod
    def fillnan(image):
        image = image
        for band in range(image.shape[0]):
            image[band] = np.nan_to_num(image[band], nan=np.nanmean(image[band]))
        return image


# ######################## S2PixelCloudDetectorWrapper ##################

class S2PixelCloudDetectorWrapper(S2PixelCloudDetector):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.MODEL_BAND_IDS = MODEL_BAND_IDS

    def check_data(self, data):
        if not data.ndim == 3:
            raise ValueError(f"This cloud detector only accepts single images with shape [band, height, width]. "
                             f"Got instead: shape={data.shape}")
        if not data.shape[0] in [13, len(self.MODEL_BAND_IDS)]:
            raise ValueError(f"This cloud detector only accepts single images with shape [band, height, width] "
                             f"with band dimension either 13 or {len(self.MODEL_BAND_IDS)}."
                             f"Got instead: shape={data.shape}, n_bands={data.shape[0]}")

    @staticmethod
    def prepare_for_cloud_detector(data):
        data = ImageUtils.bands_last(data)
        return data.clip(0, 10000) / 10000

    def get_cloud_probability_maps(self, data, **kwargs):
        """
        Cloud maps are computed and stored in the same format as used by Google Earth Engine:
        https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S2_CLOUD_PROBABILITY#description
        Namely, probabilities are rescaled to [0..100], discretized to uint8, and value 255 is used as NOVALUE mask
        """
        self.check_data(data)
        novalue_mask = ~(data[self.MODEL_BAND_IDS] != 0).prod(axis=0, dtype=bool)
        data = self.prepare_for_cloud_detector(data)

        cloud_probability_map = super().get_cloud_probability_maps(data)

        # convert probabilities to [0..100] range and discretize, e.g. convert probability to percentage
        cloud_percentage_map = np.rint(cloud_probability_map * 100).astype(np.uint8)
        cloud_percentage_map[novalue_mask] = 255
        cloud_percentage_map = cloud_percentage_map[np.newaxis, ...]
        return cloud_percentage_map

    def get_mask_from_prob(
            self,
            cloud_probs,
            threshold=None,
            average_over=None,
            dilation_size=None
    ):
        """
        Replaces the get_mask_from_prob() method of S2PixelCloudDetector with following changes:
        - allows to specify different average_over and dilation_size arguments
        - uses np.uint8 instead of np.int8
        - uses int percentages [0..100] instead of floats [0.0..1.0]

        :param cloud_probs: cloud probability map
        :type cloud_probs: numpy array of cloud probabilities (n_images, n, m) or (n, m)
        :param threshold: A float from [0..100] specifying threshold
        :param average_over: (optional) replaces the value stored in cloud detector
        :param dilation_size: (optional) replaces the value stored in cloud detector
        :type threshold: float
        :return: raster cloud mask
        :rtype: numpy array (n_images, n, m) or (n, m)
        """

        is_single_temporal = cloud_probs.ndim == 2
        if is_single_temporal:
            cloud_probs = cloud_probs[np.newaxis, ...]

        threshold, conv_filter, average_over, dilation_filter, dilation_size = self._handle_mask_parameters(
            threshold=threshold,
            average_over=average_over,
            dilation_size=dilation_size
        )

        # handle NOVALUE pixels
        novalue_masks = cloud_probs == 255
        cloud_probs[novalue_masks] = 100

        if average_over is not None:
            cloud_probs = np.asarray(
                [convolve(cloud_prob, conv_filter) for cloud_prob in cloud_probs], dtype=np.uint8
            )

        cloud_masks = (cloud_probs > threshold).astype(np.uint8)

        if dilation_size is not None:
            cloud_masks = np.asarray(
                [dilation(cloud_mask, dilation_filter) for cloud_mask in cloud_masks], dtype=np.uint8
            )
            novalue_masks = np.asarray(
                [dilation(novalue_mask, dilation_filter) for novalue_mask in novalue_masks], dtype=np.uint8
            )

        # handle NOVALUE pixels
        cloud_masks[novalue_masks] = 255

        if is_single_temporal:
            return cloud_masks.squeeze(axis=0)

        return cloud_masks

    def _handle_mask_parameters(self, threshold, average_over, dilation_size):

        threshold = self.threshold if threshold is None else threshold
        conv_filter = self.conv_filter if average_over is None else disk(average_over) / np.sum(disk(average_over))
        average_over = self.average_over if average_over is None else average_over
        dilation_filter = self.dilation_filter if dilation_size is None else disk(dilation_size)
        dilation_size = self.dilation_size if dilation_size is None else dilation_size

        return threshold, conv_filter, average_over, dilation_filter, dilation_size

