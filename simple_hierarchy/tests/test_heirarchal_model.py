import unittest

import torch
import torch.nn as nn

from simple_hierarchy.hierarchal_model import HierarchalModel
from simple_hierarchy.tree import Node, Tree

class TestHeirarchalModel(unittest.TestCase):
    def test_heirarchal_model(self):
        """
        Test that tree and layers are what is expected.
        """
        # testing tree iteration here as well
        root = Node('A', 2, None)
        child1 = Node('B', 3, root)
        child2 = Node('C', 5, root)
        child_of_child = Node('D', 3, child1)
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(child_of_child)
        tree = Tree(root)
        hierarchy = {
            ('A', 2): [('B', 3), ('C', 5)],
            ('B', 3): [('D', 3)]
        }
        model = HierarchalModel(model=nn.ModuleList([nn.Linear(10, 10) for i in range(4)]), k=2, hierarchy=hierarchy, size=(10,10,10))
        input = torch.rand((10,10))
        out = model(input)
        for t, t1 in zip(tree, model.tree):
            self.assertEqual(t.get_tuple(), t1.get_tuple())
        correct_last_layers = nn.ModuleDict({
            "('A', 2)": nn.Sequential(
                nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 2)
            ),
            "('B', 3)": nn.Sequential(
                nn.Linear(12, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 3)
            ),
            "('D', 3)": nn.Sequential(
                nn.Linear(13, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 3)
            ),
            "('C', 5)": nn.Sequential(
                nn.Linear(12, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 10),
                nn.Linear(10, 5)
            )

        })
        correct_base_model = nn.Sequential(
            nn.Linear(10, 10),
            nn.Linear(10, 10),
        )
        # compare string representations since pytorch does not have simple equality inbuilt to its classes
        self.assertEqual(str(correct_base_model), str(model.base_model))
        self.assertEqual(str(correct_last_layers), str(model.last_layers))



if __name__ == '__main__':
    unittest.main()