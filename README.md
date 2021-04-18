<p align="center"><a href="https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/hierarchy_network.jpg"><img src="https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/hierarchy_network.jpg" alt="hierarchy network" height="60"/></a></p>
<h1 align="center">simple-hierarchy</h1>
    <p align="center">Simple PyTorch hierarchical models.</p>
<p align="center">
 <a href="https://github.com/rajivsarvepalli/SimpleHierarchy/actions?workflow=Tests"><img alt="tests status" src="https://github.com/rajivsarvepalli/SimpleHierarchy/workflows/Tests/badge.svg"></a>
 <a href="https://codecov.io/gh/rajivsarvepalli/SimpleHierarchy"><img alt="codecov of simple-hierarchy" src="https://codecov.io/gh/rajivsarvepalli/SimpleHierarchy/branch/master/graph/badge.svg"></a>
 <a href="https://pypi.org/project/simple-hierarchy/"><img alt="pypi version" src="https://img.shields.io/pypi/v/simple-hierarchy.svg"></a>
 <a href="https://simplehierarchy.readthedocs.io/en/latest/?badge=latest"><img alt="most recent docs" src="https://readthedocs.org/projects/simplehierarchy/badge/?version=latest"></a>
 <a href="https://pypi.org/project/simple-hierarchy/"><img alt="supported python versions" src="https://img.shields.io/pypi/pyversions/simple-hierarchy.svg"></a>
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 <a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
 <a href="https://pepy.tech/project/simple-hierarchy"><img alt="Downloads" src="https://pepy.tech/badge/simple-hierarchy"></a>
 <a href="https://github.com/rajivsarvepalli/SimpleHierarchy/blob/master/LICENSE"><img alt="License: MIT" src="https://black.readthedocs.io/en/stable/_static/license.svg"></a>
</p><br/><br/>

# Hierarchical Classification Networks
When looking at the task for classifying something where hierarchies were intrinsic to the classes, I searched for any libraries that might do very simple classification using grouped classes with hierarchies. However, I did not find any libraries that were suited for this relatively simple task. So I sought to create a more general solution that others can hopefully benefit from.


The concept is quite simple: create general architecture for groupings of classes dependent on each other. So starting with a basic concept of model, I looked to make something in PyTorch that represented my idea.

# Example Use Case
Let us take an image geolocation problem where we want the location for city, county, and district. We will call these groupings a,b,c respectively. Given an image input, we want to predict all 3 classes but also need an architecture in which these relationships are properly represented. The network architecture below illustrates a possible solution (that this package will attempt to implement with a degree of adaptability).
The architecture can be visualized as so with an input image:
![Network Architecture](https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/network.svg)

 where the class hierarchy is like so

![Class Heirarchy](https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/tree.svg)

The class hierarchy is a similar structure to an example within this package. Each node has a tuple of a named grouping and the number of classes within that grouping. This the reason for the sizes in the final outputs in the network architecture. The large green plus signs within circles are used to indicate concatenation of the two input (green arrowed lines) leading into them. This is why the sections for class b and c have input size 4096 + 1024 = 5120.
# Installation
The required version of Python for this package is >= 3.7.

To install this package, first, install PyTorch. You can use `requirements.txt` to install PyTorch 1.7, however, the best way to install is to go to [PyTorch's website](https://pytorch.org/get-started/locally/) and follow the instructions there. This package may work with versions less than 1.7, but it was only tested on PyTorch 1.7. This package will allow for versions of PyTorch >= 1.0, but please know only 1.7 is tested.
Using pip makes this installation easy and simple once PyTorch is installed. This can be installed through
```
pip install simple-hierarchy
```
The repository can also be cloned and then built from source using poetry.

Finally, this repository can simply be downloaded and imported as python code since there are essentially only two required classes here.
# Getting Started
This architecture allows for simple yet adaptable hierarchal classifications for basic tasks that involve finite hierarchies. The package was targeted towards image classifications where there are multiple groups to classify something as but may serve other purposes equally well. Below is an example of how to use the package along with the defined class:
```
from simple_hierarchy.hierarchal_model import HierarchalModel
hierarchy = {
    ('A', 2) : [('B', 5)],
    ('B', 5) : [('C', 7)]
}
model_base = nn.Sequential(
  nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=2, stride=2),
  nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
  nn.ReLU(),
  nn.MaxPool2d(kernel_size=2, stride=2),
  nn.Flatten(start_dim=1),
  nn.Linear(in_features=1296, out_features=120),
  nn.ReLU(),
  nn.Linear(in_features=120, out_features=84),
  nn.ReLU()
)
model = HierarchalModel(hierarchy, (84, 32, 32),base_model=model_base)
# Example input
a = torch.rand(3,50,50).unsqueeze(0)
model(a)
```
Then the model can be trained on an image dataset like any other model.

Additionally, there is a [Jupyter notebook](https://github.com/rajivsarvepalli/SimpleHierarchy/blob/master/src/simple_hierarchy/examples/sample.ipynb) or [colab notebook](https://github.com/rajivsarvepalli/SimpleHierarchy/blob/master/src/simple_hierarchy/examples/sample.ipynb) within this [repository](https://github.com/rajivsarvepalli/SimpleHierarchy) that illustrates some examples of how to use and run these classes. These notebooks each contain 2 examples of how to use this package with some short explanations on what each parameter means.

The formulation is quite simple, so it should not be too much additional work to incorporate the HierarchalModel into your networks.
However, the solution given here is quite simple and therefore can be implemented easily for specific cases. The HierarchalModel class just provides a general solution for more use cases and gave me chance to test and build some architectural ideas.
## Authors

* **Rajiv Sarvepalli** - *Created* - [rajivsarvepalli](https://github.com/rajivsarvepalli)
