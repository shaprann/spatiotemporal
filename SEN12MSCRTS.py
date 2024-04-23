import numpy as np
import pandas as pd
from os import walk, makedirs
from os.path import join, isdir, dirname, abspath
import warnings
import yaml
from image import ImageFile
from tqdm import tqdm
import rasterio
from rasterio import RasterioIOError
from s2cloudless import S2PixelCloudDetector
from scipy.ndimage import gaussian_filter
from threading import Lock


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

    # define how cloud masks are calculated from cloud probability maps
    _cloud_probability_bins = np.arange(start=0.0, stop=1.0+0.05, step=0.05, dtype=np.float64)

    def __init__(
            self,
            root_dir,
            cloud_maps_dir=None,
            cloud_histograms_csv=None,
            cloud_probability_threshold=None
    ):

        if not isdir(root_dir):
            raise ValueError(f"Provided root directory does not exist: {root_dir}")
        self.root_dir = root_dir
        self.cloud_maps_dir = cloud_maps_dir
        self.cloud_histograms_csv = cloud_histograms_csv

        self._cloud_probability_threshold_index = self.locate_cloud_probability_threshold(cloud_probability_threshold)

        self._files = {}
        self._data = None
        self._cloud_histograms = None
        self._save_cloud_histograms = True

        self.cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        self.utils = ImageUtils(manager=self)

    @property
    def data(self):
        """ Getter method to prevent outer classes from editing the data """
        if self._data is None:
            raise ValueError("Dataset contains no data yet. Try running 'load_dataset()' method first.")
        return self._data.copy(deep=True)

    @property
    def cloud_histograms(self):
        """ Getter method to prevent outer classes from editing the cloud histograms DataFrame """
        if self._cloud_histograms is None:
            raise ValueError("Dataset is not initialized yet. "
                             "Try running 'load_dataset()' method first.")
        return self._cloud_histograms.copy(deep=True)

    @property
    def has_cloud_maps(self):
        return self._data is not None and "S2CLOUDMAP" in self._data

    @property
    def cloud_probability_threshold(self):
        if self._cloud_probability_threshold_index is None:
            raise ValueError("Dataset has no cloud probability threshold stored")
        return self._cloud_probability_bins[1:-1][self._cloud_probability_threshold_index]

    @property
    def cloud_probability_bins(self):
        return self._cloud_probability_bins.copy()

    def get_cloud_histogram(self, index):
        """ Getter method to get a single cloud histogram """
        if self._cloud_histograms is None:
            raise ValueError("Dataset is not initialized yet. "
                             "Try running 'load_dataset()' method first.")

        try:
            cloud_histogram = self._cloud_histograms.loc[index].copy(deep=True)
        except KeyError:
            raise ValueError("Provided index does not correspond to any index in the dataset")

        # if dataset has no cloud histogram stored, calculate it
        if cloud_histogram.isna().any():
            cloud_histogram = self.utils.calculate_cloud_histogram(index=index, store=True)

        return cloud_histogram

    def write_cloud_histogram(self, index, cloud_histogram):
        """ Setter method to store a single cloud histogram into the dataset"""

        cloud_histogram = np.array(cloud_histogram, dtype=np.int64)

        if self._cloud_histograms is None:
            raise ValueError("Can not write cloud histogram: dataset is not initialized yet. "
                             "Try running 'load_dataset()' method first.")
        if not cloud_histogram.shape == self._cloud_probability_bins[1:].shape:
            raise ValueError("Can not set cloud histogram: Provided values are incorrect")
        if np.isnan(cloud_histogram).any():
            raise ValueError("Can not set cloud histogram: Provided values contain NaNs")

        # write cloud histogram to dataset
        self._cloud_histograms.loc[index] = cloud_histogram

        # write cloud histogram to drive if necessary
        if not self._save_cloud_histograms:
            return
        elif self.cloud_histograms_csv is not None:
            self._cloud_histograms.loc[[index]].to_csv(
                self.cloud_histograms_csv,
                mode="a",
                header=False,
                float_format="%.4f"
            )

    def get_cloud_percentage(self, index):

        if self._cloud_probability_threshold_index is None:
            raise ValueError("Dataset has no cloud probability threshold stored.")

        histogram = self.get_cloud_histogram(index)
        return histogram.iloc[self._cloud_probability_threshold_index+1:].sum() / histogram.sum()

    def locate_cloud_probability_threshold(self, threshold):
        if threshold is None:
            return None
        closest_threshold = np.isclose(threshold, self._cloud_probability_bins[1:-1])  # [1:-1] --> [0.05, ..., 0.95]
        if not closest_threshold.any():
            raise ValueError(f"Threshold value {threshold:.4f} is not allowed. "
                             f"Select one of the following values instead: {self._cloud_probability_bins[1:-1]}")
        return np.argwhere(closest_threshold).item()

    def load_dataset(self):
        self.get_paths_to_files()
        self.build_dataframe()
        self.initialize_cloud_histograms()

    def load_from_file(self, filepath=None):
        if filepath is None:
            filepath = join(self.project_directory, "config", "dataset_manager.csv")
        self._data = pd.read_csv(
            filepath,
            index_col=[idx for idx in self.config["dataset_index"] if not idx == "modality"]
        )
        self.initialize_cloud_histograms()

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

    def cloud_histograms_subset(self, split=None, only_resampled=True):
        return self.get_subset(self.cloud_histograms, split=split, only_resampled=only_resampled)

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

    def initialize_cloud_histograms(self):
        self._cloud_histograms = pd.DataFrame(
            index=self.data.index,
            columns=self._cloud_probability_bins[1:],
            dtype=pd.Int64Dtype()
        )

        if self.cloud_histograms_csv is not None:
            self.connect_to_cloud_histograms_csv()

    def connect_to_cloud_histograms_csv(self):

        try:
            _ = self.data  # this throws an error if data is not initialized
            _ = self.cloud_histograms  # this throws an error if cloud histograms are not initialized
        except ValueError as error:
            raise error

        if self.cloud_histograms_csv is None:
            raise ValueError("Failed to connect to Cloud Histogram CSV: no path to CSV file is provided")

        try:
            cloud_histograms_from_csv = pd.read_csv(
                self.cloud_histograms_csv,
                index_col=[idx for idx in self.config["dataset_index"] if not idx == "modality"],
            )
            cloud_histograms_from_csv.columns = cloud_histograms_from_csv.columns.astype(np.float64)
            cloud_histograms_from_csv = cloud_histograms_from_csv.astype(pd.Int64Dtype())
            cloud_histograms_from_csv = cloud_histograms_from_csv.dropna(how="any", axis=0)
            self.check_cloud_histograms(cloud_histograms_from_csv)
            self._cloud_histograms.loc[cloud_histograms_from_csv.index] = cloud_histograms_from_csv.values
            self.save_cloud_histograms()  # this way we sort the rows in the .csv file if they were unsorted before
        except FileNotFoundError:
            warnings.warn("Failed to connect to Cloud Histograms CSV: provided CSV file does not exist. "
                          "Create a new CSV file instead.")
            self.save_cloud_histograms()  # this way we create a new .csv file

    def check_cloud_histograms(self, cloud_histograms_df):

        try:
            columns_valid = np.allclose(cloud_histograms_df.columns, self._cloud_probability_bins[1:])
        except ValueError:
            raise ValueError("Columns of Cloud Histograms CSV are not valid: wrong length")

        if not columns_valid:
            raise ValueError("Columns of Cloud Histograms CSV are not valid: wrong values")

        try:
            self.cloud_histograms.loc[cloud_histograms_df.index]
        except KeyError:
            raise ValueError("Indices of cloud Histograms CSV do not correspond to the indices in the dataset")

    def save_cloud_histograms(self):
        self.cloud_histograms.dropna(how="any", axis=0).to_csv(
            self.cloud_histograms_csv,
            float_format="%.4f"
        )

    def pause_saving_histograms(self):
        lock = CloudHistogramLock(self)
        return lock


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
            store_cloud_histogram=True
    ):

        # if index is provided, ignore cloud_map_path argument, get path from the dataset instead, and overwrite it
        if index is not None:
            try:
                cloud_map_path = self.manager.data.loc[index]["S2CLOUDMAP"]
            except KeyError:
                raise ValueError(f"Can not find path to cloud mask in the dataset using provided index: {index}")

        # if only cloud_map_path is provided, retrieve dataset index for the cloud map path
        if cloud_map_path and not index:
            index = ImageFile(manager=self.manager, filepath=cloud_map_path).short_index
            if index not in self.manager.data.index:
                raise ValueError(f"Can not find provided cloud map path in the dataset: {cloud_map_path}")

        # now we either have both index and cloud map path, or we have neither
        # if we have the cloud map path, try simply loading cloud map from the drive
        if cloud_map_path:
            try:
                cloud_map_uint16 = self.read_tif(filepath=cloud_map_path)
                cloud_map = (cloud_map_uint16 / 10000.0).astype(np.float32)  # convert from uint16 back to float32
                if store_cloud_histogram and index is not None:
                    _ = self.calculate_cloud_histogram(index=index, cloud_map=cloud_map, store=True)
                return cloud_map
            except RasterioIOError:
                pass

        # If we failed to load cloud map from drive, calculate it from S2 image and save to drive if possible

        # First, try to retrieve path to s2 image
        path_to_s2 = None
        if index:
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

        s2_image = self.prepare_for_cloud_detector(s2_image)
        cloud_map = self.manager.cloud_detector.get_cloud_probability_maps(s2_image)
        cloud_map = cloud_map[np.newaxis, ...]
        # cloud_map = gaussian_filter(cloud_map, sigma=2).astype(np.float32)

        # Try to save the newly computed cloud map to drive
        # We can only save to .tif file if we have an example Sentinel-2 .tif file,
        # so that we can copy its rasterio profile
        if cloud_map_path and path_to_s2:
            rasterio_profile = self.get_rasterio_profile(path_to_s2)

            cloud_map_uint16 = (cloud_map * 10000).astype(np.uint16)  # convert to uint16 to save space

            self.save_tif(
                filepath=cloud_map_path,
                image=cloud_map_uint16,
                rasterio_profile=rasterio_profile,
                dtype=rasterio.uint16
            )

        # Try to save cloud histogram to drive and to the dataset
        if store_cloud_histogram and index is not None:
            _ = self.calculate_cloud_histogram(index=index, cloud_map=cloud_map, store=True)

        return cloud_map

    def get_categorical_cloud_map(
            self,
            cloud_map=None,
            cloud_map_path=None,
            s2_image=None,
            index=None,
            store_cloud_histogram=True
    ):
        if cloud_map is None:
            try:
                cloud_map = self.get_cloud_map(
                    cloud_map_path=cloud_map_path,
                    s2_image=s2_image,
                    index=index,
                    store_cloud_histogram=store_cloud_histogram
                )
            except ValueError as err:
                raise err

        left_bin_edges = self.manager.cloud_probability_bins[:-1].reshape((1, 1, 1, -1))  # shape=[1,   1,   1, 20]
        right_bin_edges = self.manager.cloud_probability_bins[1:].reshape((1, 1, 1, -1))  # shape=[1,   1,   1, 20]
        cloud_map = cloud_map[..., np.newaxis]                                          # shape=[1, 256, 256,  1]

        categorical_cloud_map = (left_bin_edges <= cloud_map) & (cloud_map < right_bin_edges)

        return categorical_cloud_map                                                    # shape=[1, 256, 256, 20]

    def get_cloud_mask(
            self,
            cloud_map=None,
            cloud_map_path=None,
            s2_image=None,
            index=None,
            store_cloud_histogram=True
    ):
        try:
            # all histogram bins on the left of threshold are labeled False, on the right True
            # a bin is defined as [left, right), therefore the threshold value itself belongs to the right bin
            cloud_probability_threshold = self.manager.cloud_probability_threshold
        except ValueError as err:
            raise err

        if cloud_map is None:
            try:
                cloud_map = self.get_cloud_map(
                    cloud_map_path=cloud_map_path,
                    s2_image=s2_image,
                    index=index,
                    store_cloud_histogram=store_cloud_histogram
                )
            except ValueError as err:
                raise err

        return cloud_map >= cloud_probability_threshold  # bins are [left, right), therefore >= and not =

    def calculate_cloud_histogram(self, index, cloud_map=None, store=True):
        if cloud_map is None:
            cloud_map = self.get_cloud_map(index=index, store_cloud_histogram=False)
        cloud_histogram, _ = np.histogram(cloud_map, bins=self.manager.cloud_probability_bins, density=False)
        if store:
            self.manager.write_cloud_histogram(index, cloud_histogram)
        return cloud_histogram

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


# ####################### CloudHistogramLock ###########################

# TODO: locking cloud histograms does not work. Find out what is the problem
class CloudHistogramLock:

    def __init__(self, manager, save_step=5000):
        self.manager = manager
        self.save_step = save_step

        self.counter = 0
        self.counter_lock = Lock()

    def __enter__(self):
        self.manager._buffer_cloud_histograms = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager._buffer_cloud_histograms = None
        self.manager.save_cloud_histograms()
