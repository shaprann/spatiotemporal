import os
import rasterio
import numpy as np
from os.path import split, relpath, isfile, join
import xarray as xr
import rioxarray as rxr
import warnings

xr.set_options(keep_attrs=True)


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
        self._image = self.read_lazy()

    @property
    def image(self):
        return self._image

    @property
    def index(self):
        return self._index

    @property
    def index_string(self):
        return join(*[str(index_part) for index_part in self._index])

    def read_lazy(self):

        # TODO: rewrite hardcoded parts like timestep and patch assignment

        # read as dask array
        lazy_image = rxr.open_rasterio(self.filepath, cache=False, chunks={})

        # rename whole DataArray
        lazy_image = lazy_image.rename(self.image_type)

        # rename coordinates that were generated by rioxarray to lat, lon instead of x, y
        lazy_image = lazy_image.rename({"y": "lat", "x": "lon"})

        # add proper x and y coordinates (image coordinates)
        lazy_image = lazy_image.assign_coords(
            {
                "y": ("lat", range(lazy_image.sizes["lat"])),
                "x": ("lon", range(lazy_image.sizes["lon"]))
            }
        )

        # put bands to the end
        lazy_image = lazy_image.transpose(..., "band")

        # add timestep as dimension with coordinate
        lazy_image = lazy_image.expand_dims({"timestep": [self.metadata["timestep"]]})

        # TODO: this causes a bug: since filepath etc. is a coordinate, it is set to be the same for S1 and S2 images
        # add date, filepath, filename, and directory as coordinates which depend on timestep
        lazy_image = lazy_image.assign_coords(
            {
                "date":         ("timestep", [self.metadata["date"]]),
                "filepath":     ("timestep", [self.filepath]),
                "directory":    ("timestep", [self.directory]),
                "filename":     ("timestep", [self.filename])
            }
        )

        # add patch as coordinate
        lazy_image = lazy_image.assign_coords({"patch": self.metadata["patch"]})

        # assign new band names
        lazy_image = lazy_image.assign_coords(band=self.manager.config['bands'][self.image_type])

        # rename "bands" dimension if necessary, suppress occasional warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lazy_image = lazy_image.rename({"band": self.manager.config['band_definition'][self.image_type]})

        # set attributes
        for metadata_type, metadata in self.metadata.items():
            if metadata_type in lazy_image.coords:
                continue
            lazy_image.attrs[metadata_type] = metadata

        # does not work properly because of some strange bug in xarray :(
        # if uncommented, prevents from concatenating DataArrays together
        # lazy_image = lazy_image.set_xindex("date")

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
