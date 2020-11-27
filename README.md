# Hierarchal Classification Networks
When looking at task for classifying something where hierarchies were intrinsic to the classes, I searched for any libraries that mught do very simple classifcation using grouped classes with hierarchies. However, I did not find any libraries that were suited for this relatively simpel task. So I sought to create more general solution that others can hopefully benefit from.


The concept is quite simple: create general architecture for groupings of classes depedent on each other. So starting off with a basic concept of model, I looked to make something in pytorch that represented my idea.
The architecture can be visualized as so:
![Alt text](https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/network.svg)

 where the class heirarchy is like so

![Alt text](https://raw.githubusercontent.com/rajivsarvepalli/SimpleHierarchy/master/images/tree.svg)


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
