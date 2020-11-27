from itertools import chain
from __future__ import annotations
from typing import Tuple, Iterable

class Node(object):
    def __init__(self, name: str, n_classes: int, parent: Node) -> None:
        self.n_classes = n_classes
        self.name = name
        self.children = []
        self.parent = parent

    def add_child(self, child: Node) -> None:
        self.children.append(child)

    def __repr__(self) -> str:
        return str(self.name) + " " + str(self.n_classes) + " " + str(self.children)

    def get_tuple(self) -> Tuple[str, int]:
        return (self.name, self.n_classes)

    def __iter__(self) -> Iterable:
        def isingle(x): return (yield x)
        return chain(*([isingle(self)] + list(map(iter, self.children))))


class Tree(object):
    def __init__(self, root: Node):
        self.root = root

    def __repr__(self) -> str:
        return self.root.__repr__()

    def __iter__(self) -> Iterable:
        return iter(self.root)
