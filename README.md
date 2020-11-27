# Hierarchal Classification Networks
When looking at task for classifying something where hierarchies were intrinsic to the classes, I searched for any libraries that mught do very simple classifcation using grouped classes with hierarchies. However, I did not find any libraries that were suited for this relatively simpel task. So I sought to create more general solution that others can hopefully benefit from.


The concept is quite simple: create general architecture for groupings of classes depedent on each other. So starting off with a basic concept of model, I looked to make something in pytorch that represented my idea.

# Example
Let us take an image geolocation problem where we want the location for city, county, and districy. We will call these groupings a,b,c respectively. Given an image input, we want to predict all 3 classes but also need an architecture in which these relationships are properly represented. The network architecture below illustrates a possible solution (that this package will atttempt to implement with a degree of adaptability).
The architecture can be visualized as so with an input image:
![Network Architecture](https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/network.svg)

 where the class heirarchy is like so

![Class Heirarchy](https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/tree.svg)

The class heirarchy is a similar structure to that within the this package. Each node has tuple of a named grouping and the number of classes within that grouping. This the reason for the sizes in the final outputs in the network architecture. The large green plus signs within circles are used to indicate concatenation of the two input (green arrowed lines) leading into them. This is why the sections for class b and c have input size 4096 + 1024 = 5120.
# Installation
Using pip makes this installation easy and simple. This can be installed through 
```
pip install 
```
The repository can also be cloned and then made with pip install. This can be dones like so:
```
pip install
```
Finally, this repository can simply downloaded and imported as python code since there are essentially only two required classes here.
# Getting Started
This architecture allows for simple yet adaptable hierarchal classifications for basic tasks that involve finite hierarchies. The package was targeted towards image classifcations where there are multiple groups to classify something as, but may serve other purposes equally well. Below is an example of how to use the package along with the defined class:
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
  nn.Linear(in_features=576, out_features=120), 
  nn.ReLU(), 
  nn.Linear(in_features=120, out_features=84), 
  nn.ReLU()
)
model = HierarchalModel(hierarchy, (84, 32, 32),base_model=model_base)
```
Then the model can be trained on a image dataset like any other model.

Aditionally, there is [jupyter notebook](https://github.com/rajivsarvepalli) within this [repository](https://github.com/rajivsarvepalli) illustrates some examples of how to use and run these classes. 
The formulation is quite simple, so it should not be too much additional work to incorporate the HierarchalModel into your networks.
However, the solution given here is quite simple and therefore can be implemented easily for specific cases. The HierarchalModel class just provides a general solution for more use cases, and gave me chance to test and build some architectural ideas.   
## Authors

* **Rajiv Sarvepalli** - *Created* - [rajivsarvepalli](https://github.com/rajivsarvepalli)
