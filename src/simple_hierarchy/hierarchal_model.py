"""A module for a hierarchal model in pytorch."""
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .tree import Node, Tree


class HierarchalModel(nn.Module):
    """Creates a model for hierarchal class structures.

    Creates a model that is designed to handle hierarchal classes. It
    is targeted towards image hierarchal classification problems, but can
    be used for any finite hierarchy and network. The concept is to work
    for classes where the certain classes are children of other classes.
    For example, consider classifying cities and districts. The districts
    of city are depedent on the city classifcation. The network architecure
    is quite simple in this class's solution, nsimply take the ouput of each
    parent and feed it into the last `k` layers of all its children. In other
    words, `k` is a hyperparemeter that illusrates how many layers should be
    distinct for each class. If any of the arguments are confusing, the
    examples should help indicate how to use this class.

    Args:
        hierarchy: A defined hierarchy through a
            dictionary definition. This is defined as names for each grouping
            (arbitrary but distinct names are fine) and the number of classes
            within each group. The format is (name, n_classes) for each tuple.
            The dictionary defines the relationship between childrena and
            parents where `hierarchy[parent]` is a list of children in
            the same format as the parent which is defined above.
        size: A tuple of the (ouput size of
            the base_model or model[len(model) - k - 1], input size of
            model[len(model) - k], output size of model[len(model) - 1]. If
            `feed_from` is given then a fourth size must be given of the size of
            the `feed_from` index output size. If `join_layers` parameter is provided
            then this `size` parameter is unneeded.
        output_order: The output order of the classes
            returned by forward by their tupled keys in the hierarchy
            dictionary.
        base_model: A torch network that will serve as
            the base model.
        model: A torch module list that will be
            considered the list of network layers. If both base_model and
            this list are provided then base_model is used as base with this
            being considred for the latter layers (depending on k).
        k: A integer representing the number of last layers
            that are distinct.
        dim_to_concat: The dimension to combine parent ouput
            and the base model's ouput. Typically this is 1.
        feed_from: The reverse index to feed outputs from parent
            classes to their children. If not given, the last layer is used. For
            example, if 1 is provided than the output from the layer before the final
            layer is forwarded from parent to child. This output enters the child
            at the layers that are indepedent (determined by the `k` and `model`
            paramaters). `feed_from` must be less than `k`.
        join_layers: A dictionary with
            same keys as hierarchy dict. Each key contains a list of layers
            to join connections for that class. This is because each network
            "section" per class is join togther with layers to manage different
            sizes for different classes (since sizes are depedenet are on
            parent sizes). Without it being supplied, two linear layers
            are used per class. The first joins parent outputs into a child's
            inputs. The second ouputs into number of classes for child. If join_layers
            is provided then size is unnecessary.

    Attributes:
        base_model: The portion of the model that is the same for all classes.
        feed_from: The index to feed outputs from parent.
            classes to their children. If not given, the last layer is used.
        last_layers: A dict of the last layers of the model. Since each class has its
            own last layers, a dictionary is used. The dictionary's keys are the class
            tuples (name followed by number of classes), and the values are the layers
            for each of those classes.
        tree: The tree of the class hierarchies.
        output_order: The output order of the classes
            returned by forward by their tupled keys in the hierarchy
            dictionary.
        dim_to_concat: The dimension to combine parent ouput
            and the base model's ouput. Typically this is 1.

    Examples:
        >>> import torch.nn as nn
        >>> import torch
        >>> from simple_hierarchy.hierarchal_model import HierarchalModel
        >>> hierarchy = {("A", 2) : [("B", 5), ("C", 7)], ("H", 2) :
        >>>     [("A", 2), ("K", 7), ("L", 10)]}
        >>> base_m = nn.ModuleList([nn.Linear(10, 10) for i in range(2)])
        >>> model = HierarchalModel(model=base_m, k=1, hierarchy=hierarchy, size=(10,10,10))
        >>> len(model(torch.rand(10,10)))
        6
        >>> model.tree
        H 2 [A 2 [B 5 [], C 7 []], K 7 [], L 10 []]
    """

    def __init__(
        self,
        hierarchy: Dict[Tuple[str, int], List[Tuple[str, int]]],
        size: Optional[Tuple[int, ...]] = None,
        output_order: Optional[List] = None,
        base_model: Optional[nn.Module] = None,
        model: Optional[nn.ModuleList] = None,
        k: int = 0,
        dim_to_concat: Optional[int] = None,
        feed_from: Optional[int] = None,
        join_layers: Optional[Dict[Tuple[str, int], List[nn.Module]]] = None,
    ) -> None:
        """Creates HierarchalModel object."""
        super(HierarchalModel, self).__init__()
        if base_model:
            self.base_model = base_model
        else:
            self.base_model = nn.Sequential(*model[0 : len(model) - k])  # NOQA
        if feed_from:
            self.feed_from = feed_from + 1
            if size and len(size) != 4:
                raise ValueError(
                    "Size must contain 4 items when feed_from is provided."
                )
        else:
            self.feed_from = 0
            if size and len(size) != 3:
                raise ValueError(
                    "Size must contain 3 items when feed_from is not provided."
                )
        self.last_layers = OrderedDict()
        self.tree = self._hierarchy_to_tree(hierarchy)
        self.output_order = output_order
        if dim_to_concat:
            self.dim_to_concat = dim_to_concat
        else:
            self.dim_to_concat = 1
        for node in self.tree:
            if model:
                layer1 = model[len(model) - k : len(model)]  # NOQA
            else:
                layer1 = nn.ModuleList()
            if node.parent:
                n_classes1 = node.parent.n_classes
            else:
                n_classes1 = 0
            n_classes2 = node.n_classes
            layers = nn.ModuleList()
            if join_layers:
                j_layers = join_layers[node.get_tuple()]
                if len(j_layers) != 2:
                    raise ValueError(
                        "Each join_layers entry must have exactly two layers."
                    )
                layers.append(j_layers[0])
                layers.extend(layer1)
                layers.append(j_layers[1])
            else:
                if feed_from:
                    s1, s2, s3, s4 = size
                    if node.parent:
                        first_combiner = torch.nn.Linear(s1 + s4, s2)
                        layers.append(first_combiner)
                    else:
                        layers.append(torch.nn.Linear(s1, s2))
                    layers.extend(layer1)
                    layers.append(torch.nn.Linear(s3, n_classes2))
                else:
                    s1, s2, s3 = size
                    if node.parent:
                        first_combiner = torch.nn.Linear(s1 + n_classes1, s2)
                        layers.append(first_combiner)
                    else:
                        layers.append(torch.nn.Linear(s1, s2))
                    layers.extend(layer1)
                    layers.append(torch.nn.Linear(s3, n_classes2))

            self.last_layers[str(node.get_tuple())] = nn.Sequential(*layers)
        self.last_layers = nn.ModuleDict(self.last_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Overriding forward method for pytorch nn.Module."""
        x = self.base_model(x)
        # enumerate over a tree concating parents output into children outs
        output_temp = dict()
        output_upto_k = dict()
        for node in self.tree:
            node_key = node.get_tuple()
            if node.parent:
                parent_out = output_upto_k[node.parent.get_tuple()]
                e_in = torch.cat((parent_out, x), self.dim_to_concat)
                output_temp[node_key] = self.last_layers[str(node_key)](e_in)
                if node.children:
                    output_upto_k[node_key] = self.last_layers[str(node_key)][
                        : -self.feed_from or None
                    ](e_in)
            else:
                output_temp[node_key] = self.last_layers[str(node_key)](x)
                output_upto_k[node_key] = self.last_layers[str(node_key)][
                    : -self.feed_from or None
                ](x)
        outputs = list()
        if not self.output_order:
            self.output_order = output_temp.keys()
        for o in self.output_order:
            outputs.append(output_temp[o])
        return tuple(outputs)

    def _hierarchy_to_tree(
        self, hierarchy: Dict[Tuple[str, int], List[Tuple[str, int]]]
    ) -> Tree:
        all_children = list()
        for (_, children) in hierarchy.items():
            all_children.extend(children)
        found_root = False
        root = None
        for _, (node, _) in enumerate(hierarchy.items()):
            if node not in all_children:
                root = node
                if found_root:
                    raise ValueError("Invalid hierarchy tree.")
                found_root = True
        root_node = Node(root[0], root[1], None)
        hier = hierarchy.copy()
        self._to_tree(hier, root_node)
        return Tree(root_node)

    def _to_tree(
        self, hierarchy: Dict[Tuple[str, int], List[Tuple[str, int]]], root_node: Node
    ) -> None:
        root = root_node.get_tuple()
        for _, (node, children) in list(enumerate(hierarchy.items())):
            if root == node:
                for c in children:
                    child = Node(*c, root_node)
                    root_node.add_child(child)
                    self._to_tree(hierarchy, child)
        if root in hierarchy:
            hierarchy.pop(root)


class SimpleHierarchalModel(nn.Module):
    """Creates a simple model for hierarchal class structures.

    Creates a model that is designed to handle hierarchal classes. It
    is targeted towards image hierarchal classification problems, but can
    be used for any finite hierarchy and network. The concept is to work
    for classes where the certain classes are children of other classes.
    For example, consider classifying cities and districts. The districts
    of city are depedent on the city classifcation. The network architecure
    is quite simple in this class's solution, nsimply take the ouput of each
    parent and feed it into the last `k` layers of all its children. In other
    words, `k` is a hyperparemeter that illusrates how many layers should be
    distinct for each class. If any of the arguments are confusing, the
    examples should help indicate how to use this class.

    Args:
        hierarchy: A defined hierarchy through a
            dictionary definition. This is defined as names for each grouping
            (arbitrary but distinct names are fine) and the number of classes
            within each group. The format is (name, n_classes) for each tuple.
            The dictionary defines the relationship between childrena and
            parents where `hierarchy[parent]` is a list of children in
            the same format as the parent which is defined above.
        size: A tuple of the (ouput size of
            the base_model or model[len(model) - k - 1], input size of
            model[len(model) - k], output size of model[len(model) - 1]. If
            `feed_from` is given then a fourth size must be given of the size of
            the `feed_from` index output size. If `join_layers` parameter is provided
            then this `size` parameter is unneeded.
        output_order: The output order of the classes
            returned by forward by their tupled keys in the hierarchy
            dictionary.
        base_model: A torch network that will serve as
            the base model.
        dim_to_concat: The dimension to combine parent ouput
            and the base model's ouput. Typically this is 1.

    Attributes:
        base_model: The portion of the model that is the same for all classes.
        feed_from: The index to feed outputs from parent.
            classes to their children. If not given, the last layer is used.
        last_layers: A dict of the last layers of the model. Since each class has its
            own last layers, a dictionary is used. The dictionary's keys are the class
            tuples (name followed by number of classes), and the values are the layers
            for each of those classes.
        tree: The tree of the class hierarchies.
        output_order: The output order of the classes
            returned by forward by their tupled keys in the hierarchy
            dictionary.
        dim_to_concat: The dimension to combine parent ouput
            and the base model's ouput. Typically this is 1.

    Examples:
        >>> import torch.nn as nn
        >>> import torch
        >>> from simple_hierarchy.hierarchal_model import SimpleHierarchalModel
        >>> hierarchy = {("A", 2): [("B", 3), ("C", 5)], ("B", 3): [("D", 3)]}
        >>> base_m = nn.ModuleList([nn.Linear(10, 10) for i in range(4)])
        >>> model_b = nn.Sequential(*base_m)
        >>> model = SimpleHierarchalModel(base_model=model_b, hierarchy=hierarchy,
        >>>     size=(10,10,10))
        >>> len(model(torch.rand(10,10)))
        4
        >>> model.tree
        A 2 [B 3 [D 3 []], C 5 []]
    """

    def __init__(
        self,
        hierarchy: Dict[Tuple[str, int], List[Tuple[str, int]]],
        base_model: nn.Module,
        size: Optional[Tuple[int, ...]] = None,
        output_order: Optional[List] = None,
        dim_to_concat: Optional[int] = None,
    ) -> None:
        """Creates SimpleHierarchalModel object."""
        super(SimpleHierarchalModel, self).__init__()
        self.base_model = base_model
        self.feed_from = 0
        if size and len(size) != 3:
            raise ValueError(
                "Size must contain 3 items when feed_from is not provided."
            )
        self.last_layers = OrderedDict()
        self.tree = self._hierarchy_to_tree(hierarchy)
        self.output_order = output_order
        if dim_to_concat:
            self.dim_to_concat = dim_to_concat
        else:
            self.dim_to_concat = 1
        for node in self.tree:
            if node.parent:
                n_classes1 = node.parent.n_classes
            else:
                n_classes1 = 0
            n_classes2 = node.n_classes
            layers = nn.ModuleList()
            s1, s2, s3 = size
            if node.parent:
                first_combiner = torch.nn.Linear(s1 + n_classes1, s2)
                layers.append(first_combiner)
            else:
                layers.append(torch.nn.Linear(s1, s2))
            layers.append(torch.nn.Linear(s3, n_classes2))
            self.last_layers[str(node.get_tuple())] = nn.Sequential(*layers)
        self.last_layers = nn.ModuleDict(self.last_layers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """Overriding forward method for pytorch nn.Module."""
        x = self.base_model(x)
        # enumerate over a tree concating parents output into children outs
        output_temp = dict()
        output_upto_k = dict()
        for node in self.tree:
            node_key = node.get_tuple()
            if node.parent:
                parent_out = output_upto_k[node.parent.get_tuple()]
                e_in = torch.cat((parent_out, x), self.dim_to_concat)
                output_temp[node_key] = self.last_layers[str(node_key)](e_in)
                if node.children:
                    output_upto_k[node_key] = self.last_layers[str(node_key)][
                        : -self.feed_from or None
                    ](e_in)
            else:
                output_temp[node_key] = self.last_layers[str(node_key)](x)
                output_upto_k[node_key] = self.last_layers[str(node_key)][
                    : -self.feed_from or None
                ](x)
        outputs = list()
        if not self.output_order:
            self.output_order = output_temp.keys()
        for o in self.output_order:
            outputs.append(output_temp[o])
        return tuple(outputs)

    def _hierarchy_to_tree(
        self, hierarchy: Dict[Tuple[str, int], List[Tuple[str, int]]]
    ) -> Tree:
        all_children = list()
        for (_, children) in hierarchy.items():
            all_children.extend(children)
        found_root = False
        root = None
        for _, (node, _) in enumerate(hierarchy.items()):
            if node not in all_children:
                root = node
                if found_root:
                    raise ValueError("Invalid hierarchy tree.")
                found_root = True
        root_node = Node(root[0], root[1], None)
        hier = hierarchy.copy()
        self._to_tree(hier, root_node)
        return Tree(root_node)

    def _to_tree(
        self, hierarchy: Dict[Tuple[str, int], List[Tuple[str, int]]], root_node: Node
    ) -> None:
        root = root_node.get_tuple()
        for _, (node, children) in list(enumerate(hierarchy.items())):
            if root == node:
                for c in children:
                    child = Node(*c, root_node)
                    root_node.add_child(child)
                    self._to_tree(hierarchy, child)
        if root in hierarchy:
            hierarchy.pop(root)
