import os
import yaml
from os.path import split, abspath, dirname, join
from typing import overload, Tuple


class ImageFile:

    # get current project directory
    project_directory = abspath(dirname(__file__))

    # load dataset config file
    with open(join(project_directory, "config/sen12mscrts.yaml"), 'r') as file:
        config = yaml.safe_load(file)

    @overload
    def __init__(self, directory: str, filename: str) -> None:
        ...

    @overload
    def __init__(self, directory_filename: Tuple[str, str]) -> None:
        ...

    @overload
    def __init__(self, filepath: str) -> None:
        ...

    @overload
    def __init__(self, root_dir: str, **kwargs) -> None:
        ...

    def __init__(
            self,
            filepath=None,
            directory=None,
            filename=None,
            directory_filename=None,
            root_dir=None,
            **kwargs
    ):
        self._root_dir = None
        self._metadata = None

        if filepath is not None or directory is not None or filename is not None or directory_filename is not None:
            directory, filename = self._handle_paths(filepath, directory, filename, directory_filename)
            filename_metadata = self._parse_filename(filename=filename)
            root_dir, filepath_metadata = self._parse_filepath(directory=directory)
            self._root_dir = root_dir
            self._metadata = self._merge_metadata(filename_metadata, filepath_metadata)
        else:
            if root_dir is None:
                raise TypeError("Missing keyword argument 'root_dir'")
            self._root_dir = root_dir
            self._metadata = kwargs

        self._check_metadata()
        self._check_modality()

    def __repr__(self):
        return f"<{type(self).__name__} {self.relpath}>"

    @property
    def root_dir(self):
        return self._root_dir

    @property
    def metadata(self):
        return self._metadata.copy()

    @property
    def index(self):
        return tuple(self._metadata[name] for name in self.config['dataset_index'])

    @property
    def modality(self):
        return self._metadata["modality"]

    @property
    def index_with_modality(self):
        return *self.index, self.modality

    @property
    def filepath(self):
        return join(self.directory, self.filename)

    @property
    def relpath(self):
        return join(self.directory_relpath, self.filename)

    @property
    def directory_relpath(self):
        directory_path_content = [
            self._metadata["ROI"],
            self._metadata["tile"],
            self._metadata["modality"].upper(),
            self._metadata["timestep"]
        ]
        directory_path_content = [str(value) for value in directory_path_content]
        directory_relpath = os.sep.join(directory_path_content)

        return directory_relpath

    @property
    def directory(self):
        return join(self._root_dir, self.directory_relpath)

    @property
    def filename(self):
        filename_content = [
            self._metadata["modality"].lower(),
            self._metadata["ROI"],
            self._metadata["tile"],
            "ImgNo",
            self._metadata["timestep"],
            self._metadata["date"],
            "patch",
            self._metadata["patch"]
        ]
        filename_content = [str(value) for value in filename_content]
        filename = "_".join(filename_content) + ".tif"

        return filename

    @classmethod
    def _handle_paths(cls, filepath: str, directory: str, filename: str, directory_filename: Tuple[str, str]):

        if directory_filename is not None:

            if filepath is not None or directory is not None or filename is not None:
                raise TypeError("If you provide a 'directory_filename' tuple, "
                                "then 'directory', 'filename' and 'filepath' must be None!")
            try:
                directory, filename = directory_filename
            except ValueError as err:
                raise ValueError(f"Tuple 'directory_filename' must contain two items."
                                 f" Got instead: {directory_filename}") from err

        if filepath is not None:

            if directory is not None or filename is not None:
                raise TypeError("If you provide a 'filepath', then 'directory' and 'filename' must be None!")
            if not filepath.endswith(".tif"):
                raise ValueError(f"The 'filepath' must point to a .tif image. Got instead: {filepath}")

            directory, filename = split(filepath)

            if directory == '':
                raise ValueError(f"The 'filepath' must be an absolute path. Got instead: {filepath}")

        if directory is None or filename is None:
            raise TypeError(f"You need to provide both 'directory' and 'filename'! Got instead: "
                            f"\ndirectory: {directory}"
                            f"\nfilename: {filename}")

        return directory, filename

    @classmethod
    def _merge_metadata(cls, filepath_metadata, filename_metadata):
        cls._check_metadata_consistency(filepath_metadata, filename_metadata)
        return {**filename_metadata, **filepath_metadata}  # merge two dicts with metadata

    @classmethod
    def _check_metadata_consistency(cls, filepath_metadata, filename_metadata):
        metadata_intersection = set(filepath_metadata).intersection(set(filename_metadata))
        inconsistent_metadata = [metadata for metadata in metadata_intersection
                                 if not filepath_metadata[metadata] == filename_metadata[metadata]]
        if inconsistent_metadata:
            inconsistent_filepath_metadata = {metadata: value for metadata, value in filepath_metadata.items()
                                              if metadata in inconsistent_metadata}
            inconsistent_filename_metadata = {metadata: value for metadata, value in filename_metadata.items()
                                              if metadata in inconsistent_metadata}
            raise ValueError(f"Metadata extracted from image filename and image file path is not the same! \n"
                             f"From filename:  {inconsistent_filepath_metadata} \n"
                             f"From file path: {inconsistent_filename_metadata}")

    @classmethod
    def _parse_filename(cls, filename):
        """
        Parses information which is stored in the filename of a SEN12MS-CR-TS file.
        Expected filename format: "{modality}_{ROI}_{tile}_ImgNo_{timestep}_{date}_patch_{patch}.tif"
        Example filename:"s2_ROIs1868_100_ImgNo_9_2018-04-25_patch_17.tif"

        :return: dictionary with parsed values. Example:
            {"modality: "S2", "ROI": "ROIs1868", "tile": 100, "timestep": 9, "date": "2018-04-25", "patch": 17}
        """

        if not filename.endswith(".tif"):
            raise ValueError(f"Filename must have a .tif extension. Received instead: {filename}")

        filename = filename[:-4]  # remove .tif extension

        try:
            modality, roi, tile, _, timestep, date, _, patch = filename.split("_")  # extract information
        except ValueError as err:
            raise ValueError(f"Could not parse {cls.config['filename_metadata']} from filename {filename}")

        filename_metadata = dict(zip(
            cls.config['filename_metadata'],
            (modality.upper(), roi, int(tile), int(timestep), date, int(patch))
        ))

        return filename_metadata

    @classmethod
    def _parse_filepath(cls, directory):
        """
        Reads information which is given by file location, i.e. hierarchy of folders where file is stored.
        Example path:
            "/media/vlad/Extreme SSD/Datasets/Cloud_removal/SEN12MS-CR-TS/test/asiaWest_test/ROIs1868/100/S2/9"
        :return: dictionary with values. Example:
            {"ROI": "ROIs1868", "tile": 100, "modality: "S2", "timestep": 9}
        """

        try:
            roi, tile, modality, timestep = directory.split(os.sep)[-4:]
            root_dir = os.sep.join(directory.split(os.sep)[:-4])
        except ValueError as err:
            raise ValueError(f"Could not parse {cls.config['filepath_metadata']} "
                             f"from directory path {directory}") from err

        filepath_metadata = dict(zip(
            cls.config['filepath_metadata'],
            (roi, int(tile), modality.upper(), int(timestep))
        ))

        return root_dir, filepath_metadata

    def _check_metadata(self):
        required_metadata = set(self.config['filepath_metadata']) | set(self.config['filename_metadata'])
        if not required_metadata.issubset(self._metadata):
            raise ValueError(f"Provided metadata is not sufficient. "
                             f"Expected to get: {required_metadata}. "
                             f"Got instead: {self._metadata.keys()}")

    def _check_modality(self):
        if not self._metadata["modality"] in self.config['modalities']:
            raise ValueError(
                f"Detected a .tif image of invalid modality. "
                f"Allowed modalities in the dataset: {self.config['modalities']}. "
                f"Got instead: {self._metadata['modality']}"
            )

    def set(self, **kwargs):
        root_dir = kwargs.pop("root_dir") if "root_dir" in kwargs else self._root_dir
        metadata = {**self._metadata, **kwargs}
        return ImageFile(root_dir=root_dir, **metadata)
