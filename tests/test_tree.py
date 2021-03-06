"""Testing the tree class."""
import unittest

from simple_hierarchy.tree import Node, Tree


class TestTree(unittest.TestCase):
    """Tests the creation of tree object and its nodes."""

    def test_tree(self) -> None:
        """Test that tree inserts properly."""
        root = Node("A", 2, None)
        child1 = Node("B", 3, root)
        child2 = Node("C", 5, root)
        child_of_child = Node("D", 3, child1)
        root.add_child(child1)
        root.add_child(child2)
        child1.add_child(child_of_child)
        tree = Tree(root)
        self.assertTrue(self._is_child(root, child1))
        self.assertTrue(self._is_child(root, child2))
        self.assertTrue(self._is_child(child1, child_of_child))
        self.assertFalse(self._is_child(child_of_child, child1))
        self.assertFalse(self._is_child(child_of_child, root))
        self.assertFalse(self._is_child(child1, child2))
        correct_str = "A 2 [B 3 [D 3 []], C 5 []]"
        actual_str = str(tree)
        self.assertEqual(actual_str, correct_str)

    def _is_child(self, parent: Node, child: Node) -> bool:
        f_child = False
        for p in parent.children:
            if p.get_tuple() == child.get_tuple():
                f_child = True
                break
        if child.parent:
            return f_child and parent.get_tuple() == child.parent.get_tuple()
        else:
            return False


if __name__ == "__main__":
    unittest.main()
