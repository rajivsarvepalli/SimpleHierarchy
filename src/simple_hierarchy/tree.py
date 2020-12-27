"""A module for a tree structure for a class hierarchy."""
from __future__ import annotations  # NOQA

from itertools import chain
from typing import Iterable, Iterator, Optional, Tuple


class Node(object):
    """Stores a node with name and number of classes.

    Used to store a node with a name of number of classes. This class
    is inteded for storing class heiarchies, as such, their is both a name
    and number of classes per node. The node is linked to a list of children
    and its parent.

    Args:
        name: The name of the node.
        n_classes: The number of classes for this class that node represents.
        parent: The parent of the node, if its the root, then the parent is None.

    Attributes:
        name: The name of the node.
        n_classes: The number of classes for this class that node represents.
        parent: The parent of the node, if its the root, then the parent is None.
        children: A list of nodes that are children of this node.
    """

    def __init__(self, name: str, n_classes: int, parent: Optional[Node]) -> None:
        """Creates a Node object."""
        self.n_classes = n_classes
        self.name = name
        self.children = list()
        self.parent = parent

    def add_child(self, child: Node) -> None:
        """Adds a child node.

        Adds a child node to current node. This child is added
        to a list of children.

        Args:
            child: A node to add the current node..
        """
        self.children.append(child)

    def __repr__(self) -> str:
        """String representation of Node."""
        s = str(self.name) + " "
        s += str(self.n_classes) + " "
        s += str(self.children)
        return s

    def get_tuple(self) -> Tuple[str, int]:
        """Get tuple of name, n_classes for each node."""
        return (self.name, self.n_classes)

    def __iter__(self) -> Iterable:
        """Iterate through node and its children."""

        def _isingle(x: Node) -> Iterator[Node]:
            return (yield x)

        return chain(*([_isingle(self)] + list(map(iter, self.children))))


class Tree(object):
    """Stores a root node.

    Creates a Tree oject to store a root node as defined with the Node
    class.

    Args:
        root: The root node to store as a tree.

    Attributes:
        root: The root node of the tree.
    """

    def __init__(self, root: Node) -> None:
        """Creates a Tree object."""
        self.root = root

    def __repr__(self) -> str:
        """String representation of Tree."""
        return self.root.__repr__()

    def __iter__(self) -> Iterable:
        """Iterate through root of node."""
        return iter(self.root)
