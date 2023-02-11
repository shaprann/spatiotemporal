import numpy as np
import pandas as pd
import os
from os.path import join, isfile, isdir
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
    min_max = pd.read_csv(join(project_directory, "98_percentile_min_max.csv"), index_col="band")

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

        # self.load_dataset()
        # self.load_cloudmasks()
        # self.load_cloud_percentage()

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


    def get_cloud_map(self, cloud_map_path, s2_image):

        threshold = 0.5

        try:
            return self.read_tif(cloud_map_path)

        except RasterioIOError:
            if type(s2_image) is str:
                s2_image = self.read_tif(s2_image)
            cloud_map = self.cloud_detector.get_cloud_probability_maps(s2_image)
            cloud_map = cloud_map[np.newaxis, ...]
            cloud_map[cloud_map < threshold] = 0
            cloud_map = gaussian_filter(cloud_map, sigma=2).astype(np.float32)
            return cloud_map

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

    @staticmethod
    def read_tif(filepath):
        return rasterio.open(filepath).read()

    @classmethod
    def bands_last(cls, s2_image):
        return np.transpose(s2_image, (1, 2, 0))

    @classmethod
    def prepare_for_cloud_detector(cls, s2_image):
        s2_image = cls.bands_last(s2_image)
        return s2_image.clip(0, 10000) / 10000

    @classmethod
    def rescale_s2(cls, s2_image):

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