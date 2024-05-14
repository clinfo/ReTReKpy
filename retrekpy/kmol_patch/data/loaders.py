""" The ``retrekpy.kmol_patch.data`` package ``loaders`` module. """

from typing import List, Union

from kmol.data.loaders import AbstractLoader, DataPoint


class PatchedListLoader(AbstractLoader):
    """ The patched 'ListLoader' class. """

    def __init__(
            self,
            data: List[DataPoint],
            indices: List[str]
    ) -> None:
        """ The patched constructor method of the class. """

        self._dataset = data
        self._indices = indices
        self._map_indices = dict(zip(indices, range(len(indices))))

    def __len__(
            self
    ) -> int:
        """ The patched '__len__' method  of the class. """

        return len(self._dataset)

    def __getitem__(
            self,
            id_: str
    ) -> DataPoint:
        """ The patched '__getitem__' method of the class. """

        return self._dataset[self._map_indices[id_]]

    def list_ids(
            self
    ) -> List[Union[int, str]]:
        """ The patched 'list_ids' method of the class. """

        return self._indices

    def get_labels(
            self
    ) -> List[str]:
        """ The patched 'get_labels' method of the class. """

        return self._dataset[0].labels
