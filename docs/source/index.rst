.. Crowd_counting documentation master file, created by
   sphinx-quickstart on Wed May  4 15:38:13 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Crowd counting's documentation!
==========================================

Crowd_counting is a library to train and use your own crowd counting models using python3 and based on tensorflow. It was developped in the context of the master Thesis of Louis Robins and Henri Collin at UCLouvain.
The source code is available on `github <https://github.com/lrobins1/crowd_counting>`_ .

The aim of this documentation is to guide anyone that would like to 

* Train a crowd counting model based on the CSRNet architecture on its own datas
* Use a pretrain model to predict a density map and the number of people on an image

The CSRNet model implementation is highly inspired from the `implementation of the creators <https://github.com/leeyeehoo/CSRNet-pytorch>`_ . Some changes were made to adapt it to Python 3. 

.. toctree::
   :maxdepth: 2
   :caption: Guides:

   install 
   new_train
   al_train


   
.. toctree::
   :maxdepth: 2
   :caption: Modules:
   
   augmentation
   ground_truth
   train
   model
   
   

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
