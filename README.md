# TSPCS-Net
(This work was submitted to Engineering Applications of Artificial Intelligence, and now it is revising.)
# Introduction about this Repository
1. dataset folder: place your datasets. 
2. logs folder: record some training information after every epoch during the period of training.
3. networks folder: place your network model. Here, TSPCS_Net.py is our model.
4. test_results folder: save outputs when testing the well-trained model on test set.
5. weights folder: save the best model and its parameters during the period of training.
6. Constants.py: set some hyper-parameters, e.g., epochs, which dataset used, classes, etc.
7. data.py: generate dataloader for training, including how to read images and labels for different datasets.
8. framework.py: code some information about training networks.
9. loss.py: set the loss candidates.
10. test_TSPCS-Net.py: test the trained model on test set.
11. train_TSPCS-Net.py: train our network model on training set.

# Used Python and Pytorch Versions
Python 3.8, Pytorch 1.12.1, one GPU

