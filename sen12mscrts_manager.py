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
import pickle
import dask

xr.set_options(keep_attrs=True)


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
        if not isdir(root_dir):
            raise ValueError(f"Provided root directory does not exist: {root_dir}")
        self.root_dir = root_dir
        self.cloud_masks_dir = cloud_masks_dir
        self.cloud_percentage_csv = cloud_percentage_csv

        # create a dask.delayed function which can be applied on DataArrays to generate lazy cloud maps
        # ensure that cloud detector is copied to dask graph only once, and not every time cloud map is computed
        delayed_detector_class = dask.delayed(S2PixelCloudDetector)
        delayed_detector_instance = delayed_detector_class(threshold=0.4, all_bands=True, average_over=4, dilation_size=2)
        self.cloud_map_function = delayed_detector_instance.get_cloud_probability_maps

        self._data_found = []
        self._data = None

        # self.load_dataset()

    @property
    def data(self):
        """ Getter method to prevent outer classes from editing the registry """
        return self._data

    def load_dataset(self):

        self.find_and_read_files()
        self.build_tree()

        print("Add cloud maps and cloud percentages...")
        self.add_cloud_maps()
        self.add_cloud_percentages()

        self.merge_by_timestep()

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
                    self._data_found.append(image_reader)

    def build_tree(self):

        dt = datatree.DataTree(name="SEN12MS-CR-TS")
        for image_reader in tqdm(self._data_found, desc="Add images to DataTree"):
            dt[image_reader.index_string] = image_reader.image
        self._data = dt
        self._data_found = []

    # The function is ugly, but it will do for now
    def merge_by_timestep(self):

        # TODO: replace by len(self.leaves) in future versions of DataTree
        n_items = len([node for node in self._data.subtree if node.is_leaf])

        with tqdm(total=n_items, desc="Merge images over timestep") as pbar:

            for roi, roi_tree in self._data.children.items():
                for tile, tile_tree in roi_tree.children.items():
                    for patch, patch_tree in tile_tree.children.items():

                        try:
                            patch_tree.ds = xr.combine_by_coords(
                                [node.to_dataset() for node in patch_tree.values()],
                                combine_attrs="drop_conflicts"
                            )

                            pbar.update(len(patch_tree))

                            del patch_tree.children

                        except ValueError:
                            pbar.write(f"Found faulty patch {patch} at {roi} in tile {tile}\n"
                                       f"Remove tile {roi}/{tile} from the dataset entirely.")

                            pbar.update(len(patch_tree) * len(tile_tree))

                            # detach current tile subtree from the .data tree
                            tile_tree.parent = None

                            # release current tile subtree (just in case)
                            del tile_tree

                            # break from the patch loop. The tile loop will continue with the next tile
                            break

    def add_cloud_maps(self):
        self._data = self._data.map_over_subtree(self.add_cloud_map)

    def add_cloud_map(self, dataset):

        # dataset could be a datatree.DatasetView, therefore we explicitly cast it to xr.Dataset
        dataset = (xr.Dataset)(dataset)

        s2_data = dataset["S2"].data / 10000
        delayed_cloud_map = self.cloud_map_function(s2_data)

        # prepare a placeholder array which will hold the cloud map
        dataset["S2_cloud_map"] = dataset["S2"].isel(band=0).astype(float)

        # delayed_cloud_map is a dask.Delayed object, not a true dask array
        # therefore we need an ad-hoc conversion to dask array first
        delayed_array = dask.array.from_delayed(
            delayed_cloud_map,
            dataset["S2_cloud_map"].shape,
            dataset["S2_cloud_map"].dtype
        )

        dataset["S2_cloud_map"].data = delayed_array

        return dataset

    def add_cloud_percentages(self):
        self._data = self._data.map_over_subtree(self.add_cloud_percentage)

    @classmethod
    def add_cloud_percentage(cls, dataset):
        dataset = (xr.Dataset)(dataset)
        dataset["S2_cloud_percentage"] = dataset["S2_cloud_map"].mean(dim=["lat", "lon"])
        return dataset

    def save(self, filepath=None):

        if filepath is None:
            filepath = join(self.root_dir, "sen12mscrts_datatree.pickle")

        with open(filepath, "wb") as file:
            pickle.dump(self._data, file)

    def load_from_file(self, filepath=None):

        if filepath is None:
            filepath = join(self.root_dir, "sen12mscrts_datatree.pickle")

        with open(filepath, 'rb') as f:
            self._data = pickle.load(f)

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
