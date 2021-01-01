"""Testing the hierarchical model."""
import unittest

import torch
import torch.nn as nn

from simple_hierarchy.hierarchal_model import HierarchalModel
from simple_hierarchy.tree import Node, Tree


class TestHeirarchalModel(unittest.TestCase):
    """Tests the heirarchal model creation and output computation."""

    def test_heirarchal_model(self) -> None:
        """Test that tree and layers are what is expected."""
        # testing tree iteration here as well
        root = Node("A", 2, None)
        child1 = Node("B", 3, root)
        child2 = Node("C", 5, root)
        child_of_child = Node("D", 3, child1)
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(child_of_child)
        tree = Tree(root)
        hierarchy = {("A", 2): [("B", 3), ("C", 5)], ("B", 3): [("D", 3)]}
        model_b = nn.ModuleList([nn.Linear(10, 10) for i in range(4)])
        model = HierarchalModel(
            model=model_b, k=2, hierarchy=hierarchy, size=(10, 10, 10)
        )
        for t, t1 in zip(tree, model.tree):
            self.assertEqual(t.get_tuple(), t1.get_tuple())
        correct_last_layers = nn.ModuleDict(
            {
                "('A', 2)": nn.Sequential(
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 2),
                ),
                "('B', 3)": nn.Sequential(
                    nn.Linear(12, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 3),
                ),
                "('D', 3)": nn.Sequential(
                    nn.Linear(13, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 3),
                ),
                "('C', 5)": nn.Sequential(
                    nn.Linear(12, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 10),
                    nn.Linear(10, 5),
                ),
            }
        )
        correct_base_model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        )
        # compare string representations since pytorch
        # does not have simple equality inbuilt to its classes
        self.assertEqual(str(correct_base_model), str(model.base_model))
        self.assertEqual(str(correct_last_layers), str(model.last_layers))
        a = torch.rand(10, 10)
        out = model(a)
        self.assertEqual(len(out), 4)
        sizes_out = [(10, 2), (10, 3), (10, 3), (10, 5)]
        for s, out in zip(sizes_out, out):
            self.assertEqual(s, out.shape)

    def test_heirarchal_model_v2(self) -> None:
        """Tests features of choosing when to split inputs and where to feed from."""
        root = Node("A", 2, None)
        child1 = Node("B", 3, root)
        child2 = Node("C", 5, root)
        child_of_child = Node("D", 3, child1)
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(child_of_child)
        tree = Tree(root)
        hierarchy = {("A", 2): [("B", 3), ("C", 5)], ("B", 3): [("D", 3)]}
        model_b = nn.ModuleList(
            [
                nn.Linear(10, 10),
                nn.Linear(10, 40),
                nn.Linear(40, 90),
                nn.Linear(50, 20),
                nn.Linear(20, 10),
            ]
        )
        join_layers = {
            ("A", 2): [nn.Linear(90, 50), nn.Linear(10, 2)],
            ("B", 3): [nn.Linear(110, 50), nn.Linear(10, 3)],
            ("D", 3): [nn.Linear(110, 50), nn.Linear(10, 3)],
            ("C", 5): [nn.Linear(110, 50), nn.Linear(10, 5)],
        }
        output_order = [("A", 2), ("B", 3), ("D", 3), ("C", 5)]
        model = HierarchalModel(
            model=model_b,
            k=2,
            hierarchy=hierarchy,
            size=None,
            feed_from=1,
            join_layers=join_layers,
            dim_to_concat=1,
            output_order=output_order,
        )
        for t, t1 in zip(tree, model.tree):
            self.assertEqual(t.get_tuple(), t1.get_tuple())
        correct_last_layers = nn.ModuleDict(
            {
                "('A', 2)": nn.Sequential(
                    nn.Linear(90, 50),
                    nn.Linear(50, 20),
                    nn.Linear(20, 10),
                    nn.Linear(10, 2),
                ),
                "('B', 3)": nn.Sequential(
                    nn.Linear(110, 50),
                    nn.Linear(50, 20),
                    nn.Linear(20, 10),
                    nn.Linear(10, 3),
                ),
                "('D', 3)": nn.Sequential(
                    nn.Linear(110, 50),
                    nn.Linear(50, 20),
                    nn.Linear(20, 10),
                    nn.Linear(10, 3),
                ),
                "('C', 5)": nn.Sequential(
                    nn.Linear(110, 50),
                    nn.Linear(50, 20),
                    nn.Linear(20, 10),
                    nn.Linear(10, 5),
                ),
            }
        )
        correct_base_model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 40),
            nn.Linear(40, 90),
        )
        # compare string representations since pytorch
        # does not have simple equality inbuilt to its classes
        self.assertEqual(str(correct_base_model), str(model.base_model))
        self.assertEqual(str(correct_last_layers), str(model.last_layers))
        a = torch.rand(10, 10)
        model(a)
        # only using base example
        model_b = nn.ModuleList(
            [
                nn.Linear(10, 10),
                nn.Linear(10, 40),
                nn.Linear(40, 90),
                nn.Linear(90, 20),
                nn.Linear(20, 10),
            ]
        )
        model = HierarchalModel(
            base_model=nn.Sequential(*model_b), hierarchy=hierarchy, size=(10, 10, 10)
        )
        a = torch.rand(10, 10)
        model(a)

        model_b = nn.ModuleList(
            [
                nn.Linear(10, 10),
                nn.Linear(10, 40),
                nn.Linear(40, 90),
                nn.Linear(50, 20),
                nn.Linear(20, 10),
            ]
        )
        model = HierarchalModel(
            model=model_b,
            k=2,
            hierarchy=hierarchy,
            size=(90, 50, 10, 20),
            feed_from=1,
            dim_to_concat=1,
        )
        a = torch.rand(10, 10)
        model(a)

    def test_heirarchal_model_exceptions(self) -> None:
        """Tests exceptions are raised properly."""
        hierarchy = {("A", 2): [("B", 3), ("C", 5)], ("B", 3): [("D", 3)]}
        model_b = nn.ModuleList([nn.Linear(10, 10) for i in range(4)])
        # invalid size too small
        with self.assertRaises(ValueError):
            _ = HierarchalModel(
                model=model_b, k=2, hierarchy=hierarchy, size=(10, 10, 10), feed_from=1
            )
        # invalid size too big
        with self.assertRaises(ValueError):
            _ = HierarchalModel(
                model=model_b, k=2, hierarchy=hierarchy, size=(10, 10, 10, 10)
            )
        join_layers = {
            ("A", 2): [nn.Linear(90, 50)],
            ("B", 3): [nn.Linear(110, 50), nn.Linear(10, 3)],
            ("D", 3): [nn.Linear(110, 50), nn.Linear(10, 3)],
            ("C", 5): [nn.Linear(110, 50), nn.Linear(10, 5)],
        }
        # invalid join layers
        with self.assertRaises(ValueError):
            _ = HierarchalModel(
                model=model_b, k=2, hierarchy=hierarchy, join_layers=join_layers
            )
        # invalid hierarchy
        hierarchy = {("A", 2): [("C", 5)], ("B", 3): [("D", 3)]}
        with self.assertRaises(ValueError):
            _ = HierarchalModel(
                model=model_b, k=2, hierarchy=hierarchy, size=(10, 10, 10)
            )


if __name__ == "__main__":
    unittest.main()
