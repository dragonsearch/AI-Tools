Some code snippets and recurring classes that are useful as a baseline for other projects. 
- Training.py includes a class that handles the training process in a modular way so it is easier to modify and read instead of having everything in one single file.
- Evaluate.py does something similar as training but for evaluation. Having those as 2 separate classes makes it easier to customize in case we want to do different things in training vs eval.
- Utils.py is just a collection of functions such as setting seeds, showing images, simple plots or saving/loading objects/models.

Requirements are torch>=2.1, numpy>=1.26, and matplotlib>=3.8.0 torchmetrics>= 1.2, python>=3.10.13 but should be compatible with almost every version as long as its somewhat recent.
