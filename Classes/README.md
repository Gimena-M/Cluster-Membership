This directory contains the following classes:
* DataHandler: handles data reading and preparing.
* ModelTrainer: trains a model. Superclass, not meant to be instantiated. Inherits to NNModelTrainer and RFModelTrainer.
* ModelTester: tests a model. Superclass, not meant to be instantiated. Inherits to NNModelTester and RFModelTester.
* NNModelController: creates NNModelTrainer and NNModelTester instances, and manages DataHandler.

Classes can be imported in scripts with:

    import sys
    sys.path.append('../../Classes') # with the relative path to this directory
    from DataHandler import DataHandler

The .py files for each class contain a brief summary of their usage.