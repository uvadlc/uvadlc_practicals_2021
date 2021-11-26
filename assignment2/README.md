# Assignment 2: CNNS, RNNs, and GNNs

The assignment is organized in three parts. The first part deals with CNNs, where you test to robustness of different architectures on some corruption functions. In the second part, you implement your own LSTM and train it on a text dataset. Finally, in part 3, you implement a small GNN that you train on a molecular dataset. 

## Part 1: CNNs
* We provide a tiny model `debug` which you can use for debugging your code. Make sure that the training and testing works on your local machine before testing it on Lisa. 
* When submitting the code, __make sure to not include the trained models and/or CIFAR10 dataset__.

## Part 2: LSTMs
* We provide a set of possible text datasets in the folder `assets`. However, you are not limited to those and can add your own. 

## Part 3: GNNs
* The implementation of the GNN requires the package PyTorch Geometric. You can install this package via Conda. For instance, on Lisa you can install PyTorch Geometric in the dl2021 environment as follows:
```bash
conda install pyg -c pyg -c conda-forge
```
* A installation guide can be found in the [PyTorch documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html). If you have troubles installing it, please ask one of your TAs.
* The file `permute.pt` saves a particular permutation of the data for training/validation/test. This makes sure that everyone works on the same data split.
* You might see the following warning: `Using a pre-processed version of the dataset. Please install 'rdkit' to alternatively process the raw data.` You do not need to install rdkit and can use the data as is.
