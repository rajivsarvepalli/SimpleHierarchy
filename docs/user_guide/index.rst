.. _user_guide:

User Guide
==========

There several major uses for this package and here I will try to detail out several examples.

For the first example, we will consider an example problem. We want to classify an image to a geographical location for which we have latitude and longitude.
Given this problem, we can form hierarchal clusters of our geocoordinates. Using this, we have a set of classifications (a grouping of classes). In this problem,
let us classify it into 3 groupings or 3 sets of clusters termed a, b, c. Cluster grouping a is the parent of cluster grouping b and cluster grouping b is the parent of cluster grouping c.
Within the cluster grouping of a, we have 3 options or classes, Within the cluster grouping of b, there are 5 options, and cluster grouping c has 10 options.

In a real-world scenario, we might consider this to something like cluster grouping a is countries, cluster grouping b is states or territories, and cluster grouping c is counties.
We then want to classify images based into these categories, since it makes more sense to be able to classify some degree of location. Therefore, we create groupings of things that are easier
to predict with children groupings that are of decreasing scope and increasing difficulty. Below is a tree of the hierarchy described:

.. image:: https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/tree.svg
   :target: https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/tree.svg


We can construct a model where the last 4 layers are independent (as in they are different for every grouping of classes) and the second to last layer feeds forward from each parent to its child.
The reason we do not construct a model per class grouping is due to the expensiveness of doing so. Additionally, we assume that groupings of classes have a high amount of shared information, therefore,
providing some amount of shared information is useful. Therefore, we model our network after what we might consider the information to share. There is a decent amount of sharing between child and parent, so
our network shares the same base. However, these groupings of classes have differences so there are a set of independent layers. The network also has these characteristics since the last :code:`k` layers are
independent. Finally, using PyTorch, we can simply output multiple predictions. However, this method fails to have independent layers per class grouping. Essentially, this is the reason I created this library:
to make a network that fits those criteria. For an illustration of this network like the one described above, see the below image.

.. image:: https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/network.svg
   :target: https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/network.svg

.. _user_guide_examples:

Example Models
==============

.. _user_guide_example_1:

Example 1
---------

Let us walk through an example of how to construct a similar code using :mod:`simple_hierarchy`.

First, let us create the hierarchy.

.. code-block:: python

   hierarchy = {
      ("A", 3) : [("B", 5)],
      ("B", 5): [("C", 10)]
   }

Now let's create our PyTorch model class.

.. code-block:: python

   import torch.nn as nn
   class DemoModel(nn.Module):
      def __init__(self, hierarchy, base_model, size, model_layers, k, feed_from, output_order):
         super(DemoModel, self).__init__()
         self.model = HierarchalModel(
                           base_model=base_model,
                           hierarchy=hierarchy,
                           size=size,
                           model=model_layers,
                           k=k,
                           feed_from=feed_from,
                           output_order=output_order,
                     )
      def forward(self, x):
         return self.model(x)

We now have a PyTorch model class that can everything that a normal PyTorch model can do, except it outputs three items.
The three items are outputted in the order according to the :code:`output_order` variable we provide, but they will contain
the predictions for class grouping a, class grouping b, and class grouping c.

Now let us create the model base and independent layers. Independent layers refer to the layers that are not shared or separate for
child and parent classes.

.. code-block:: python

   import torch
   import torchvision
   import torch.nn as nn
   # use GPU if available in PyTorch
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   # these are the indepdent layers of parent and children
   model_layers = [
      nn.Linear(800, 750),
      nn.Linear(750, 512),
      # the output of this layer is feed forward from parent to child
      nn.Linear(512, 128),
      nn.Linear(128, 64),
   ]
   base_model = torchvision.models.resnext101_32x8d(pretrained = True)
   # 1000 is the output size of our base model (the resnext101_32x8d)
   # 800 is the input size of our additional indepdent layers (called model_layers)
   # 64 is the output size of our additional indepdent layers (called model_layers)
   # 128 is the output size of second to last additional indepdent layer to feed
   # forward from parent to child (with concatenation)
   size = (1000,800,64,128)
   # all 4 layers are distinct for each grouping of classes of model_layers
   k = 4
   # we want to feed from the second to last layer (from parent to child (with concatenation))
   feed_from = 1
   output_order = [("A", 3), ("B", 5), ("C", 10)]
   model = DemoModel(hierarchy, base_model, size, model_layers, k, feed_from, output_order)
   model = model.to(device)

Now we can train the model as we please just like any other PyTorch model. Now this model feeds from the output of the second to last layer (:code:`nn.Linear(128, 64)`) of each parent into the
4th to last layer of the child (:code:`nn.Linear(800, 750)`). What is meant by feeds forwards is that the base model output for the child is concatenated with the parent output from the second to last
layer. The :class:`simple_hierarchy.hierarchal_model.HierarchalModel` handles this through adding an additional Linear layer to manage
the sizes due to inputs being larger for children nodes since we concatenate the parent outputs with the child inputs. Let us run the model on a random input and see the outputs shapes:

.. code-block:: python

   out = torch.rand((8, 3, 512, 512))
   pred = model(out)
   for p in pred:
      print(p.shape)

In which you should get output on the console of something like:

.. code-block:: console

   torch.Size([8, 3])
   torch.Size([8, 5])
   torch.Size([8, 10])

The outputs are of the expected size due to the batch size being 8, and the other number is the respective number of classes in class grouping a, b, and c.
You can run softmax on these outputs to get the predicted classifications.


Example 2
---------

The second example will include a less detailed explanation (go to the :ref:`first example <user_guide_example_1>` to see a more detailed explanation of how the library functions),
but will illustrate how to configure :code:`join_layers` for more configurable means of connecting parent outputs and child inputs. This could even be used for the combination
of more complex layers, but here is a simpler example.

.. code-block:: python

   import torch
   import torch.nn as nn
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
   a = torch.rand(10, 10)
   out = model(a)

The model output will be contained inside the :code:`out` variable.


Additionally, there is a `Jupyter notebook <https://github.com/rajivsarvepalli/SimpleHierarchy/blob/master/src/simple_hierarchy/examples/sample.ipynb>`__
includes several examples.
