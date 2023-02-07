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

        # self.load_dataset()
        # self.load_cloudmasks()
        # self.load_cloud_percentage()

    @property
    def data(self):
        """ Getter method to prevent outer classes from editing the registry """
        return self._data

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

                    image_reader = ImageReader(manager=self, directory=current_path, filename=filename)

                    self._files[image_reader.index] = image_reader.filepath

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

    def add_paths_to_cloudmasks(self):
        """Deprecated! Do not use"""

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
