#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


import numpy as np

from torch.utils.data import Dataset

from abc import abstractmethod
from scipy.spatial import KDTree
from numbers import Number

from autoreconstruction.pytorch_segment.neurotorch.datasets.datatypes import (
    BoundingBox, Vector)


class Data:
    """
    An encapsulating object for communicating volumetric data
    """
    def __init__(
            self,
            array: np.ndarray,
            bounding_box: BoundingBox
    ):
        """
        Initializes a data packet from an Numpy array and its bounding box

        :param array: A Numpy array containing the data packet's contents
        :param bounding_box: A bounding box specifying the data packet's
            location in 3D-space
        """
        self.bounding_box = bounding_box
        self.array = array  # row-major order (Z, Y, X)

    def __add__(
            self,
            other
    ):
        assert isinstance(other, Data), "other must have type Data"
        assert self.bounding_box == other.bounding_box, (
            "other must have the same bounding box")

        return Data(self.array + other.array, self.bounding_box)

    def __sub__(
            self,
            other
    ):
        return self + (-other)

    def __neg__(
            self
    ):
        return (self * -1)

    def __mul__(
            self,
            other
    ):
        assert isinstance(other, Number), (
            f"other must be a number instead it is a {type(other)}")

        return Data(self.array * other, self.bounding_box)

    def __div__(
            self,
            other
    ):
        assert isinstance(other, Number), (
            f"other must be a number instead it is a {type(other)}")

        return self * ( 1 /other)


class Array:
    """
    A dataset containing a 3D volumetric array
    """
    def __init__(
            self,
            array: np.ndarray,
            bounding_box: BoundingBox = None,
            iteration_size: BoundingBox = BoundingBox(
                Vector(0, 0, 0), Vector(128, 128, 32)),
            stride: Vector = Vector(64, 64, 16)
    ):
        """
        Initializes a volume with a bounding box and iteration parameters

        :param array: A 3D Numpy array
        :param bounding_box: The bounding box encompassing the volume
        :param iteration_size: The bounding box of each data sample in the
            dataset iterable
        :param stride: The stride displacement of each data sample in the
            dataset iterable. The displacement proceeds first from X then to Y
            then to Z.
        """
        # assert type(array) in (np.ndarray, BoundingBox), (
        #     "array must be an ndarray or a BoundingBox")

        self.array = array
        self.bounding_box = (
            BoundingBox(Vector(0, 0, 0), Vector(*self.array.shape[::-1]))
            if bounding_box is None else bounding_box)
        self.setIteration(iteration_size=iteration_size,
                          stride=stride)
        super().__init__()

    def get(self, bounding_box: BoundingBox) -> Data:
        """
        Requests a data sample from the volume. If the bounding box does
not exist, then the method raises a ValueError.

        :param bounding_box: The bounding box of the request data sample
        :return: The data sample requested
        """
        if bounding_box.is_disjoint(self.bounding_box):
            error_string = ("Bounding box must be inside dataset " +
                            "dimensions instead bounding box is {} while " +
                            "the dataset dimensions are {}")
            error_string = error_string.format(bounding_box,
                                               self.bounding_box)
            raise ValueError(error_string)

        sub_bounding_box = bounding_box.intersect(self.bounding_box)
        array = self.getArray(sub_bounding_box)

        before_pad = bounding_box.get_edges()[0] - sub_bounding_box.get_edges()[0]
        after_pad = bounding_box.get_edges()[1] - sub_bounding_box.get_edges()[1]

        if before_pad != Vector(0, 0, 0) or after_pad != Vector(0, 0, 0):
            pad_size = tuple(zip(before_pad.numpy_dims,
                                 after_pad.numpy_dims))
            array = np.pad(array, pad_width=pad_size, mode="constant")

        return Data(array, bounding_box)

    def set(self, data: Data):
        """
        Sets a section of the volume within the provided bounding box with the
given data.

        :param data: The data packet to set the volume
        """
        data_bounding_box = data.bounding_box
        data_array = data.array

        if not data_bounding_box.is_subset(self.bounding_box):
            raise ValueError("The bounding box must be a subset of the "
                             " volume")

        data_edge1, data_edge2 = data_bounding_box.get_edges()
        array_edge1, array_edge2 = self.bounding_box.get_edges()

        edge1 = data_edge1 - array_edge1
        edge2 = data_edge2 - array_edge1

        x1, y1, z1 = edge1.components
        x2, y2, z2 = edge2.components

        self.array[z1:z2, y1:y2, x1:x2] = data_array

    def blend(self, data: Data):
        """
        Blends a section of the volume within the provided bounding box with
the given data by taking the elementwise maximum value.

        :param data: The data packet to blend into the volume
        """
        array = self.get(data.bounding_box).array
        array = np.maximum(array, data.array)

        result = Data(array, data.bounding_box)

        self.set(result)

    def getArray(
            self,
            bounding_box: BoundingBox = None
    ) -> np.ndarray:
        """
        Retrieves the array contents of the volume. If a bounding box is
provided, the subsection is returned.

        :param bounding_box: The bounding box of a subsection of the volume.
If the bounding box is outside of the volume, a ValueError is raised.
        """
        if bounding_box is None:
            return self.array
        else:
            if not bounding_box.is_subset(self.bounding_box):
                raise ValueError("The bounding box must be a subset" +
                                 " of the volume")

            centered_bounding_box = bounding_box - self.bounding_box.get_edges()[0]
            edge1, edge2 = centered_bounding_box.get_edges()
            x1, y1, z1 = edge1.components
            x2, y2, z2 = edge2.components

            return self.array[z1:z2, y1:y2, x1:x2]

    def setIteration(self, iteration_size: BoundingBox, stride: Vector):
        """
        Sets the parameters for iterating through the dataset

        :param iteration_size: The size of each data sample in the volume
        :param stride: The displacement of each iteration
        """
        if not isinstance(iteration_size, BoundingBox):
            error_string = ("iteration_size must have type BoundingBox"
                            + " instead it has type {}")
            error_string = error_string.format(type(iteration_size))
            raise ValueError(error_string)

        if not isinstance(stride, Vector):
            raise ValueError("stride must have type Vector")

        if not iteration_size.is_subset(BoundingBox(Vector(0, 0, 0),
                                                    self.bounding_box.size)):
            raise ValueError("iteration_size must be smaller than volume size")

        self.setIterationSize(iteration_size)
        self.setStride(stride)

        def ceil(x):
            return int(round(x))

        self.element_vec = Vector(*map(lambda L, l, s: ceil((L-l)/s+1),
                                       self.bounding_box.size.components,
                                       self.iteration_size.size.components,
                                       self.stride.components))

        self.index = 0

    def setIterationSize(self, iteration_size):
        self.iteration_size = BoundingBox(Vector(0, 0, 0),
                                          iteration_size.size)

    def setStride(self, stride):
        self.stride = stride

    def getIterationSize(self):
        return self.iteration_size

    def getStride(self):
        return self.stride

    def __len__(self):
        return self.element_vec[0]*self.element_vec[1]*self.element_vec[2]

    def __getitem__(self, idx):
        bounding_box = self._indexToBoundingBox(idx)
        result = self.get(bounding_box)

        return result

    def _indexToBoundingBox(self, idx):
        if idx >= len(self):
            self.index = 0
            raise StopIteration

        element_vec = np.unravel_index(idx, shape=self.element_vec.components)

        element_vec = Vector(*element_vec)
        bounding_box = self.iteration_size+self.stride*element_vec

        return bounding_box

    def __enter__(self):
        pass

    def __exit__(self):
        pass


class TorchVolume(Dataset):
    def __init__(self, volume):
        self.setVolume(volume)
        super().__init__()

    def __len__(self):
        return len(self.getVolume())

    def __getitem__(self, idx):
        if isinstance(self.getVolume(), AlignedVolume):
            data_list = [self.toTorch(data) for data in self.getVolume()[idx]]
            return data_list
        else:
            return self.getVolume()[idx].array

    def toTorch(self, data):
        torch_data = data.array.astype(np.float)
        torch_data = torch_data.reshape(1, *torch_data.shape)
        return torch_data

    def setVolume(self, volume):
        self.volume = volume

    def getVolume(self):
        return self.volume


class Volume:
    """
    An interface for creating volumes
    """
    def __init__(self, bounding_box: BoundingBox=None,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 32)),
                 stride: Vector=Vector(64, 64, 16)):
        self.setBoundingBox(bounding_box)
        self.setIteration(iteration_size, stride)
        self.valid_data = None

    def setArray(self, array: Array):
        self.array = array

    def getArray(self) -> Array:
        return self.array

    def request(self, bounding_box):
        return self.array.get(bounding_box)

    def set(self, data: Data):
        self.array.set(data)

    def blend(self, data: Data):
        self.array.blend(data)

    def setIteration(self, iteration_size: BoundingBox, stride: Vector):
        if not isinstance(iteration_size, BoundingBox):
            error_string = ("iteration_size must have type BoundingBox"
                            + " instead it has type {}")
            error_string = error_string.format(type(iteration_size))
            raise ValueError(error_string)

        if not isinstance(stride, Vector):
            raise ValueError("stride must have type Vector")

        if not iteration_size.is_subset(BoundingBox(Vector(0, 0, 0),
                                                    self.getBoundingBox().size)):
            raise ValueError("iteration_size must be smaller than volume size " +
                             "instead the iteration size is {} ".format(iteration_size.size) +
                             "and the volume size is {}".format(self.getBoundingBox().size))

        self.setIterationSize(iteration_size)
        self.setStride(stride)

        def ceil(x):
            return int(round(x))

        self.element_vec = Vector(*map(lambda L, l, s: ceil((L-l)/s+1),
                                       self.getBoundingBox().size.components,
                                       self.iteration_size.size.components,
                                       self.stride.components))

        self.index = 0

    def setBoundingBox(self, bounding_box):
        if not isinstance(bounding_box, BoundingBox):
            raise ValueError("bounding_box must have type BoundingBox " +
                             "instead it has type {}".format(type(bounding_box)))
        self.bounding_box = bounding_box

    def getBoundingBox(self):
        return self.bounding_box

    def setIterationSize(self, iteration_size):
        if not isinstance(iteration_size, BoundingBox):
            raise ValueError("iteration_size must have type BoundingBox")
        self.iteration_size = iteration_size

    def getIterationSize(self):
        return self.iteration_size

    def setStride(self, stride):
        if not isinstance(stride, Vector):
            raise ValueError("stride must have type Vector")
        self.stride = stride

    def getStride(self):
        return self.stride

    @abstractmethod
    def loadArray(self):
        pass

    @abstractmethod
    def unloadArray(self):
        pass

    @abstractmethod
    def get(self, bounding_box: BoundingBox) -> Data:
        """
        Requests a data sample from the dataset. If the bounding box does
not exist, then the method raises a ValueError.

        :param bounding_box: The bounding box of the request data sample
        :return: The data sample requested
        """
        pass

    @abstractmethod
    def set(self, data: Data):
        """
        Sets a section of the dataset within the provided bounding box with the
given data.

        :param data: The data packet to set the volume
        """
        pass

    @abstractmethod
    def __enter__(self):
        """
        Loads the dataset into memory
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_value, traceback):
        """
        Unloads the dataset from memory
        """
        pass

    def __len__(self) -> int:
        """
        Returns the length of the dataset

        :return: The dataset length
        """
        return self.element_vec[0]*self.element_vec[1]*self.element_vec[2]

    def __getitem__(self, idx: int):
        """
        Returns the data sample at index idx from the dataset

        :param idx: The index of the data sample
        """
        return self.array[idx]

    def __iter__(self):
        """
        Returns an iterable of the dataset

        :return: The iterable of the dataset
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Retrieves the next data sample from the dataset
        :return: The next data sample
        """
        if self.index < len(self):
            result = self.__getitem__(self.index)
            self.index += 1
            return result
        else:
            raise StopIteration

    def getValidData(self):
        if self.valid_data is not None:
            self.valid_data = []
            for i in range(len(self)):
                if not (self[i].array == 0).all():
                    self.valid_data.append(i)

        return self.valid_data


class AlignedVolume(Volume):
    def __init__(self, volumes, iteration_size=None, stride=None):
        if iteration_size is None:
            iteration_size = volumes[0].getIterationSize()
        if stride is None:
            stride = volumes[0].getStride()
        self.setVolumes(volumes)
        self.setIteration(iteration_size, stride)
        self.valid_data = None

    def getBoundingBox(self):
        return self.getVolumes()[0].bounding_box

    def setVolumes(self, volumes):
        self.volumes = volumes

    def addVolume(self, volume):
        self.volumes.append(volume)

    def getVolumes(self):
        return self.volumes

    def setIteration(self, iteration_size, stride):
        for volume in self.getVolumes():
            volume.setIteration(iteration_size, stride)

    def get(self, bounding_box):
        result = [volume.get(bounding_box)
                  for volume in self.getVolumes()]
        return result

    def set(self, array, bounding_box):
        pass

    def __len__(self):
        return len(self.getVolumes()[0])

    def __getitem__(self, idx):
        result = [volume[idx] for volume in self.getVolumes()]
        return result

    def getValidData(self):
        if self.valid_data is None:
            self.valid_data = []
            for i in range(len(self)):
                if not (self.getVolumes()[1][i].array == 0).all():
                    self.valid_data.append(i)

        return self.valid_data

    def _indexToBoundingBox(self, idx):
        bounding_box = self.getVolumes()[0]._indexToBoundingBox(idx)

        return bounding_box


class PooledVolume(Volume):
    def __init__(self, volumes=None, stack_size: int=5,
                 iteration_size: BoundingBox=BoundingBox(Vector(0, 0, 0),
                                                         Vector(128, 128, 32)),
                 stride: Vector=Vector(64, 64, 16)):
        if volumes is not None:
            self.volumes = volumes
            self.volumes_changed = True
        else:
            self.volumes = []
            self.volumes_changed = False

        self.volume_list = []
        self.setStack(stack_size)

        self.setIteration(iteration_size, stride)

        self.valid_data = None

    def setStack(self, stack_size: int=15):
        self.stack = []
        self.stack_size = stack_size

    def _pushStack(self, index, volume):
        if len(self.stack) >= self.stack_size:
            self.stack[0][1].__exit__(None, None, None)
            self.stack.pop(0)

        pos = len(self.stack)
        self.stack.insert(pos, (index, volume.__enter__()))

        return pos

    def _rebuildIndexes(self):
        edge1_list = [volume.bounding_box.get_edges()[0].components
                      for volume in self.volumes]

        self.edge1_list = KDTree(edge1_list)

        self.__len__()

        self.volumes_changed = False

    def _queryBoundingBox(self, bounding_box: BoundingBox) -> Volume:
        if self.volumes_changed:
            self._rebuildIndexes()

        edge1 = [bounding_box.get_edges()[0].components]
        distances, indexes  = self.edge1_list.query(edge1, k=8)
        indexes = [index for index, dist in zip(indexes[0], distances[0])
                   if dist < float('Inf')]
        indexes = filter(lambda index: not bounding_box.is_disjoint(self.volumes[index].getBoundingBox()),
                         indexes)
        if not indexes:
            raise IndexError("bounding_box is not present in any indexes")

        return list(indexes)

    def add(self, volume: Volume):
        self.volumes_changed = True
        self.volumes.append(volume)

    def get(self, bounding_box: BoundingBox) -> Data:
        indexes = self._queryBoundingBox(bounding_box)

        data = []

        stack_volumes = [volume for i, volume in self.stack if i in indexes]
        stack_disjoint = list(set(indexes) - set([i for i, v in self.stack]))

        for volume in stack_volumes:
            sub_bbox = bounding_box.intersect(volume.bounding_box)
            data.append(volume.get(sub_bbox))

        for index in stack_disjoint:
            volume = self.volumes[index]
            i = self._pushStack(index, volume)

            sub_bbox = bounding_box.intersect(volume.bounding_box)
            data.append(volume.get(sub_bbox))

        shape = bounding_box.numpy_dims
        array = Array(np.zeros(shape).astype(np.uint16),
                        bounding_box=bounding_box,
                        iteration_size=BoundingBox(Vector(0, 0, 0),
                                                    bounding_box.size),
                        stride=bounding_box.size)
        [array.set(item) for item in data]
        return Data(array.array, bounding_box)

    def set(self, data: Data):
        indexes = self._queryBoundingBox(data.bounding_box)

        data = []
        for index in indexes:
            for stack_index, stack_volume in self.stack:
                if stack_index == index:
                    stack_volume.set(data)
                else:
                    volume = self.volume_list[index].__enter__()
                    self._pushStack(index, volume)

                    volume.set(data)

    def __exit__(self, exc_type, exc_value, traceback):
        for index, volume in self.stack:
            volume.__exit__()

    def __len__(self) -> int:
        if self.volumes_changed:
            self.volume_index = [0]
            for volume in self.volumes:
                self.volume_index.append(self.volume_index[-1] + len(volume))
            self.length = self.volume_index[-1]

        return self.length

    def __getitem__(self, idx: int) -> Data:
        bounding_box = self._indexToBoundingBox(idx)
        result = self.get(bounding_box)

        return result

    def _indexToBoundingBox(self, idx: int) -> BoundingBox:
        if self.volumes_changed:
            len(self)

        if idx >= len(self):
            self.index = 0
            raise StopIteration

        index = max(filter(lambda index: self.volume_index[index] <= idx,
                           range(len(self.volume_index))))
        volume = self.volumes[index]
        _idx = idx-self.volume_index[index]

        element_vec = np.unravel_index(_idx,
                                       dims=volume.element_vec.components)

        element_vec = Vector(*element_vec)
        bounding_box = volume.iteration_size+volume.stride*element_vec \
                       + volume.bounding_box.edge1

        return bounding_box

    def setIteration(self, iteration_size: BoundingBox, stride: Vector):
        for volume in self.volume_list:
            volume.setIteration(iteration_size, stride)

        self.setIterationSize(iteration_size)
        self.setStride(stride)

    def getValidData(self):
        if self.valid_data is None:
            self.valid_data = []
            for i in range(len(self)):
                if not (self[i].array == 0).all():
                    self.valid_data.append(i)

        return self.valid_data
