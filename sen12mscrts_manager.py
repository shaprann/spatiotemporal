import pandas as pd
import os
from os.path import join, isfile, isdir
import warnings
import yaml
from image import ImageReader
from tqdm import tqdm
import xarray as xr
import datatree
from s2cloudless import S2PixelCloudDetector

xr.set_options(keep_attrs=True)


class Sen12mscrtsDatasetManager:
    """
    File manager which finds SEN12MS-CR-TS dataset files, parses some metadata, and stores it in the registry.
    Registry is stored as a MultiIndex pd.DataFrame under the .registry variable.
    """

    with open("sen12mscrts.yaml", 'r') as file:
        config = yaml.safe_load(file)

    cloud_detector = S2PixelCloudDetector(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)

    def __init__(
            self,
            root_dir,
            cloud_masks_dir=None,
            cloud_percentage_csv=None,
            cloud_percentage_buffer=None
    ):
        if not isdir(root_dir):
            raise ValueError(f"Provided root directory does not exist: {root_dir}")
        self.root_dir = root_dir
        self.cloud_masks_dir = cloud_masks_dir
        self.cloud_percentage_csv = cloud_percentage_csv

        self._data_found = {}
        self._data = None

        # self.load_dataset()

    @property
    def data(self):
        """ Getter method to prevent outer classes from editing the registry """
        return self._data

    def load_dataset(self):

        self.find_and_read_files()
        self.build_tree()
        self.merge_by_timestep()

        print("Add cloud masks...")
        self.add_cloud_masks()

        print("Add cloud percentage...")
        self.add_cloud_percentages()

        print("Done!")

    def find_and_read_files(self):
        """
        Creates a pd.DataFrame, finds all dataset files in root directory, and adds them to the dataframe
        :return:
        """

        n_files = sum([len(files) for r, d, files in os.walk(self.root_dir)])
        with tqdm(total=n_files, desc="Load dataset files lazily") as pbar:

            for current_path, directories, filenames in os.walk(self.root_dir):

                for filename in filenames:

                    pbar.update(1)

                    # skip everything except for .tif images
                    if not filename.endswith(".tif"):
                        continue

                    image_reader = ImageReader(
                        manager=self,
                        directory=current_path,
                        filename=filename
                    )
                    self._data_found[image_reader.index_string] = image_reader.image

    def build_tree(self):

        dt = datatree.DataTree(name="SEN12MS-CR-TS")
        for index_string, image in tqdm(self._data_found.items(), desc="Add images to DataTree"):
            dt[index_string] = image
        self._data = dt

    # TODO: write a more pythonic function which follows sen12mscrts.yaml specifications
    def merge_by_timestep(self):

        # TODO: replace by len(self.leaves) in future versions of DataTree
        n_items = len([node for node in self._data.subtree if node.is_leaf])

        with tqdm(total=n_items, desc="Merge images over timestep") as pbar:

            for roi, roi_tree in self._data.children.items():
                for tile, tile_tree in roi_tree.children.items():
                    for patch, patch_tree in tile_tree.children.items():
                        pbar.update(len(patch_tree))
                        patch_tree.ds = xr.combine_by_coords(
                            [node.to_dataset() for node in patch_tree.values()],
                            combine_attrs="drop_conflicts"
                        )
                        del patch_tree.children

    def add_cloud_masks(self):
        self._data = self._data.map_over_subtree(self.add_cloud_mask)

    @classmethod
    def add_cloud_mask(cls, dataset) -> xr.Dataset:

        dataset = (xr.Dataset)(dataset)
        s2 = dataset["S2"]
        s2 = s2.clip(min=0, max=10000, keep_attrs=True)
        s2 = s2 / 10000

        dataset["cloud_mask"] = xr.apply_ufunc(
            cls.cloud_detector.get_cloud_probability_maps,
            s2,
            input_core_dims=[["lat", "lon", "band"]],
            output_core_dims=[["lat", "lon"]],
            exclude_dims=set(("band",)),
            dask='parallelized',
            output_dtypes=[s2.dtype]
        )
        return dataset

    def add_cloud_percentages(self):
        self._data = self._data.map_over_subtree(self.add_cloud_percentage)

    @classmethod
    def add_cloud_percentage(cls, dataset):
        dataset = (xr.Dataset)(dataset)
        dataset["cloud_percentage"] = dataset["cloud_mask"].mean(dim=["lat", "lon"])
        return dataset

    def add_path_to_cloudmask(self):

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
