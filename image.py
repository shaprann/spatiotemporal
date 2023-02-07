import os
import rasterio
import numpy as np
from os.path import split, relpath, isfile, isdir, join
import xarray as xr
import rioxarray as rxr
import warnings


class ImageReader:

    def __init__(
            self,
            manager,
            filepath=None,
            directory=None,
            filename=None
    ):
        self.manager = manager
        self.filepath, self.directory, self.filename = self._handle_paths(filepath, directory, filename)

        self.filename_metadata = self._parse_filename()
        self.filepath_metadata = self._parse_file_location()
        self._check_metadata_consistency()
        self.metadata = {**self.filename_metadata, **self.filepath_metadata}  # merge two dicts with metadata

        self.image_type = self._detect_image_type()
        self._index = self._detect_index()
        self._image = None

    @property
    def image(self):
        return self._image

    @property
    def index(self):
        return self._index

    @property
    def optical(self):
        return True if self.image_type == "S2" else False

    def read_lazy(self):
        """Deprecated"""

        # TODO: rewrite hardcoded parts like timestep and patch assignment

        # read as dask array
        lazy_image = rxr.open_rasterio(self.filepath, cache=False, chunks={})

        # rename DataArray
        lazy_image = lazy_image.rename(self.image_type)

        # add timestep as dimension with coordinate
        lazy_image = lazy_image.expand_dims({"timestep": [self.metadata["timestep"]]})

        # add 'date' as coordinate, create index so that we can use it to select data.
        lazy_image = lazy_image.assign_coords({"date": ("timestep", [self.metadata["date"]])})

        # add patch as coordinate
        lazy_image = lazy_image.assign_coords({"patch": self.metadata["patch"]})

        # assign new band names
        lazy_image = lazy_image.assign_coords(band=self.manager.config['bands'][self.image_type])

        # rename the coordinate for bands if necessary, suppress occasional warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lazy_image = lazy_image.rename({"band": self.manager.config['band_definition'][self.image_type]})

        # set attributes
        for metadata_type, metadata in self.metadata.items():
            if metadata_type in lazy_image.coords:
                continue
            lazy_image.attrs[metadata_type] = metadata

        # add an index to "date", so that we can use it to select data
        # doing it earlier in the code does not work because of some strange bug
        # lazy_image = lazy_image.set_xindex("date")

        lazy_image = lazy_image.to_dataset()

        return lazy_image

    @classmethod
    def _handle_paths(cls, filepath, directory, filename):
        if filepath:
            directory, filename = split(filepath)
        elif directory and filename:
            filepath = join(directory, filename)
        else:
            raise ValueError("Either a full image path or a combimation of folder and filename must be provided")
        return filepath, directory, filename

    def _parse_filename(self):
        """
        Parses information which is stored in the filename of a SEN12MS-CR-TS file.
        Expected filename format: "{modality}_{ROI}_{tile}_ImgNo_{timestep}_{date}_patch_{patch}.tif"
        Example filename:"s2_ROIs1868_100_ImgNo_9_2018-04-25_patch_17.tif"
        :param str filename: filename to parse
        :return: dictionary with values. Example:
            {"modality: "S2", "ROI": "ROIs1868", "tile": 100, "timestep": 9, "date": "2018-04-25", "patch": 17}
        """

        if not self.filename.endswith(".tif"):
            raise ValueError(f"Filename must have a .tif extension. Received instead: {self.filename}")

        filename = self.filename[:-4]  # remove .tif extension

        try:
            modality, roi, tile, _, timestep, date, _, patch = filename.split("_")  # extract information
        except ValueError as err:
            raise ValueError(f"Could not parse {self.manager.config['filename_contains']} from filename {filename}")

        return dict(zip(
            self.manager.config['filename_contains'],
            (modality.upper(), roi, int(tile), int(timestep), date, int(patch))
        ))

    def _parse_file_location(self):
        """
        Reads information which is given by file location, i.e. hierarchy of folders where file is stored.
        :param str directory: full path to folder where the file is stored, without filename itself. Example:
            "/media/vlad/Extreme SSD/Datasets/Cloud_removal/SEN12MS-CR-TS/test/asiaWest_test/ROIs1868/100/S2/9"
        :return: dictionary with values. Example:
            {"ROI": "ROIs1868", "tile": 100, "modality: "S2", "timestep": 9}
        """

        if not self.directory.startswith(self.manager.root_dir):
            raise ValueError(f"Expected a path from root directory. Received instead: {self.directory}")

        # convert to relative path (removes root_dir from the path)
        directory = relpath(self.directory, self.manager.root_dir)

        try:
            roi, tile, modality, timestep = directory.split(os.sep)
        except ValueError as err:
            raise ValueError(f"Could not parse {self.manager.config['hierarchy_in_storage']} from directory path {directory}")

        return dict(zip(
            self.manager.config['hierarchy_in_storage'],
            (roi, int(tile), modality, int(timestep))
        ))

    def _check_metadata_consistency(self):
        same_metadata = [content for content in self.filepath_metadata if content in self.filename_metadata]
        for metadata in same_metadata:
            if not self.filepath_metadata[metadata] == self.filename_metadata[metadata]:
                raise ValueError(f"Metadata extracted from image filename and image location is not same! \n"
                                 f"Filename {metadata} is {self.filename_metadata[metadata]} \n"
                                 f"Location {metadata} is {self.filepath_metadata[metadata]}")

    def _detect_image_type(self):
        if self.metadata["modality"] in self.manager.config['dataset_content']:
            return self.metadata["modality"]
        else:
            raise ValueError(f"Can not detect image type. "
                             f"Allowed types: {self.manager.config['dataset_content']}. "
                             f"Got instead: {self.metadata['modality']}")

    def _detect_index(self):
        return tuple(self.metadata[index_level] for index_level in self.manager.config['dataset_index'])

    def check_cloud_map_prerequisites(self):

        if not self.optical:
            raise ValueError(f"Can only generate cloud maps for optical images. Got instead: {self.image_type}")

        if not self.manager.cloud_maps_dir:
            raise ValueError("Unable to read or save cloud maps: path to cloud maps was not provided.")

        if not isdir(self.manager.cloud_maps_dir):
            raise ValueError(f"Unable to read or save cloud maps. "
                             f"{self.manager.cloud_maps_dir} is not a valid directory.")

    @property
    def cloud_map_index(self):
        self.check_cloud_map_prerequisites()
        index = list(self._index)
        index[-1] = "S2_cloud_map"
        return tuple(index)

    @property
    def path_to_cloud_map(self):
        self.check_cloud_map_prerequisites()

        # here we use string join
        filename_content = [
            "s2cloudmap",
            self.metadata["ROI"],
            self.metadata["tile"],
            "ImgNo",
            self.metadata["timestep"],
            self.metadata["date"],
            "patch",
            self.metadata["patch"]
        ]
        filename = "_".join([str(value) for value in filename_content]) + ".tif"

        # here we use os.path.join
        directory_path_content = [
            self.manager.cloud_maps_dir,
            self.metadata["ROI"],
            self.metadata["tile"],
            self.metadata["modality"],
            self.metadata["timestep"]
        ]
        directory = join(*[str(value) for value in directory_path_content])

        return join(directory, filename)
