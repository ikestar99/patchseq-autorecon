#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thur Sep 19 09:00:00 2024
@origin: https://github.com/ogliko/patchseq-autorecon
"""


from numbers import Number


class Vector:
    """
    A basic vector data type
    """
    def __init__(
            self,
            *args: Number
    ):
        """
        Initializes a vector
        :param args: Numbers specifying the components of the vector
        :type args: List of numbers
        """
        assert all(isinstance(x, Number) for x in args)
        self.components = tuple(args)
        # self.dimensions = len(self.components)
        self.numpy_dims = self.components[::-1]

    def __len__(
            self
    ):
        return len(self.components)

    def __getitem__(
            self,
            idx: int
    ):
        return self.components[idx]

    def __add__(
            self,
            other
    ):
        if isinstance(other, Number):
            result = [s + other for s in self.components]
        elif isinstance(other, Vector):
            assert len(self) == len(other), (
                f"dimension error: {len(self)} and {len(other)}")
            result = [s + o for s, o in zip(self.components, other.components)]
        else:
            raise ValueError(
                f"other must be a number or a vector, not {type(other)}")

        return Vector(*result)

    def __mul__(
            self,
            other
    ):
        if isinstance(other, Number):
            result = [s * other for s in self.components]
        elif isinstance(other, Vector):
            result = [
                s * o for s, o in zip(self.components, other.components)]
        else:
            raise ValueError(
                f"other must be a number or a vector, not {type(other)}")

        return Vector(*result)

    def __div__(
            self,
            other
    ):
        if isinstance(other, Number):
            return self * (1 / other)
        elif isinstance(other, Vector):
            return self * Vector(*[1 / o for o in other.components])
        else:
            raise ValueError(
                f"other must be a number or a vector, not {type(other)}")

    def __sub__(
            self,
            other
    ):
        return self + (other * -1)

    def __neg__(
            self
    ):
        return self * -1

    def __eq__(
            self,
            other
    ):
        assert isinstance(other, Vector), (
            f"other must be a vector, not {type(other)}")

        return self.components == other.components

    def __ne__(
            self,
            other
    ):
        return not self == other

    def __str__(
            self
    ):
        return f"{self.components}"

    def __iter__(
            self
    ):
        return iter(self.components)


class BoundingBox:
    """
    A basic data type specifying a cube
    """
    def __init__(
            self,
            edge1: Vector,
            edge2: Vector
    ):
        """
        Initializes a bounding box
        :param edge1: A vector specifying the first edge of the box
        :param edge2: A vector specifying the second edge of the box
        """
        assert all((isinstance(edge1, Vector), isinstance(edge2, Vector))), (
            "edges must be vectors")
        assert edge1.dimensions == edge2.dimensions, (
            f"edges mismatched, edge 1: {edge1} and edge 2: {edge2}")

        self.edge1 = edge1
        self.edge2 = edge2
        self.size = edge2 - edge1
        self.numpy_dims = self.size.components[::-1]

    def get_edges(
            self
    ):
        """
        Returns the two edges for the bounding box
        """
        return self.edge1, self.edge2

    def is_disjoint(
            self,
            other
    ):
        """
        Determines whether two bounding boxes are disjoint from each other
        :param other: The other bounding box for comparison
        :type other: BoundingBox
        :return: True if the bounding boxes are disjoint, false otherwise
        :rtype: bool
        """
        if not isinstance(other, BoundingBox):
            raise ValueError("other must be a vector instead other is "
                             "{}".format(type(other)))

        # result = any(r1 > r2 for r1, r2 in zip(self.get_edges()[0],
        #                                        other.get_edges()[1]))
        # result |= any(r1 < r2 for r1, r2 in zip(self.get_edges()[1],
        #                                         other.get_edges()[0]))
        return any(
            [se1 > oe2 for se1, oe2 in zip(self.edge1, other.edge2)] +
            [oe1 > se2 for se2, oe1 in zip(self.edge2, other.edge1)])

    def is_subset(
            self,
            other
    ):
        """
        Determines whether the bounding box is a subset of the other
        :param other: The other bounding box for comparison
        :type other: BoundingBox
        :return: True if the bounding box is a subset of the other, false
            otherwise
        :rtype: bool
        """
        assert isinstance(other, BoundingBox), (
            f"other must be a vector instead other is {type(other)}")

        # result = any(r1 <= r2 for r1, r2 in zip(self.get_edges()[1],
        #                                         other.get_edges()[1]))
        # result &= any(r1 >= r2 for r1, r2 in zip(self.get_edges()[0],
        #                                          other.get_edges()[0]))

        # Determines whether the first bounding box's components are
        # within the other's components
        return all(
            [se2 <= oe2 for se2, oe2 in zip(self.edge2, other.edeg2)] +
            [se1 >= oe1 for se1, oe1 in zip(self.edge1, other.edge1)])

    def intersect(
            self,
            other
    ):
        """
        Returns the bounding box given by the intersection of two
        bounding boxes
        :param other: The other bounding box for the intersection
        :type other: BoundingBox
        :return: An bounding box intersecting the two bounding boxes
        :rtype: BoundingBox
        """
        assert not self.is_disjoint(other), "Bounding boxes cannot be disjoint"

        # largest components of first edges
        edge1 = Vector(*[
            max(self.edge1.components[i], other.edge1.components[i])
            for i in range(len(self.edge1))])

        # smallest components of second edges
        edge2 = Vector(*[
            min(self.edge2.components[i], other.edge2.components[i])
            for i in range(len(self.edge1))])
        return BoundingBox(edge1, edge2)

    def __str__(
            self
    ):
        return f"Edge1: {self.edge1}\nEdge2: {self.edge2}"

    def __add__(
            self,
            other
    ):
        assert isinstance(other, Vector), (
                f"other must be a vector instead other is {type(other)}")

        return BoundingBox(self.edge1 + other, self.edge2 + other)

    def __sub__(
            self,
            other
    ):
        return self + (other * -1)

    def __eq__(
            self,
            other
    ):
        assert isinstance(other, BoundingBox), "other must be a BoundingBox"

        return (self.edge1 == other.edge1) and (self.edge2 == other.edge2)

    def __ne__(
            self,
            other
    ):
        return not self == other
