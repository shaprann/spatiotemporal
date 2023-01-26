import pandas as pd
import os
from os.path import join, isfile, isdir
import warnings
import yaml
from image import ImageReader
from tqdm import tqdm


class Sen12mscrtsDatasetManager:
    """
    File manager which finds SEN12MS-CR-TS dataset files, parses some metadata, and stores it in the registry.
    Registry is stored as a MultiIndex pd.DataFrame under the .registry variable.
    """

    with open("sen12mscrts.yaml", 'r') as file:
        config = yaml.safe_load(file)

    def __init__(
            self,
            root_dir,
            cloud_masks_dir=None,
            cloud_percentage_csv=None,
            cloud_percentage_buffer=None
    ):

        self.root_dir = root_dir
        self.cloud_masks_dir = cloud_masks_dir
        self.cloud_percentage_csv = cloud_percentage_csv
        self._data_found = {}
        self._data = None

        # self.load_dataset()
        # self.load_cloudmasks()
        # self.load_cloud_percentage()

    @property
    def data(self):
        """ Getter method to prevent outer classes from editing the registry """
        return self._data

    def load_dataset(self):
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

                    image_reader = ImageReader(manager=self, dir_path=current_path, filename=filename)

                    if image_reader.index not in self._data_found:
                        self._data_found[image_reader.index] = image_reader.image
                    else:
                        self._data_found[image_reader.index].update(image_reader.image)

        self.build_dataframe()

    def build_dataframe(self):

        # create an empty series
        self._data = pd.Series(
            index=pd.MultiIndex.from_tuples(self._data_found.keys(), names=self.config["dataset_index"]),
            dtype="object"
        )

        # manually fill with data, because pandas will not allow it otherwise
        for index, image in self._data_found.items():
            self._data.at[index] = image

        # sort
        self._data = self._data.sort_index()

        # convert to dataframe
        self._data = self._data.to_frame(name="xarray")

    def add_paths_to_cloudmasks(self):

        if not self.cloud_masks_dir:
            warnings.warn("Unable to read or save cloud masks: path to cloud masks was not provided. ")
            return

        if not isdir(self.cloud_masks_dir):
            warnings.warn(f"Unable to read or save cloud masks. {self.cloud_masks_dir} is not a valid directory.")
            return

        self._data["S2_cloudmask_filename"] = (
            "s2cloudmask_"
            + self._data.index.get_level_values("ROI")
            + "_"
            + self._data.index.get_level_values("tile").astype(str)
            + "_ImgNo_"
            + self._data.index.get_level_values("timestep").astype(str)
            + "_"
            + self._data["S2_date"]
            + "_patch_"
            + self._data.index.get_level_values("patch").astype(str)
            + ".tif"
        )

        self._data["S2_cloudmask_abspath"] = (
            self.cloud_masks_dir
            + os.sep
            + self._data.index.get_level_values("ROI").astype(str)
            + os.sep
            + self._data.index.get_level_values("tile").astype(str)
            + os.sep
            + "S2"
            + os.sep
            + self._data.index.get_level_values("timestep").astype(str)
            + os.sep
            + self._data["S2_cloudmask_filename"]
        )

        self._data["S2_cloudmask_exists"] = self._data["S2_cloudmask_abspath"].map(isfile)
