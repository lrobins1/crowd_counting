Train a new model
===================

Folder structure
******************
To ensure a smooth runnig of all the function during the process, the folder containing the images should be organised as the folowing structure.
The images should have the format .jpeg or .png, and the ground_truth should have the .mat format.
The .mat files can be generated based on the images by using this `Matlab script <https://github.com/princenarula222/Crowd_Annotation>`_

| Project
| ├── images
| │   └── all images
| │   
| └── ground_truth
|     └── all ground_truth
| 

The first step is then to load a list of the path to all your images using augentation's dir_to_list function. 
This list will be used in a lot of the following steps.

.. code-block:: python
    
    from Crowd_counting.augmentation import dir_to_list
    img_path = 'Project/images'
    img_paths,GT_paths = dir_to_list(img_path)
    

Data augmentation
******************
The :ref:`rst_augment` permits to perform some data augmentation directly on the images and the ground truth.
Several modifications are available.


Ground_truth generation
************************
In order to train the models, it is first needed to create the ground_truth density maps in the .h5 format as in the following code.

.. code-block:: python
    
    from Crowd_counting.ground_truth import gt_gen
    gt_gen(img_paths,Verbose = True)


Model training
****************
The model is trainable directly in python.

.. code-block:: python 
     
    from Crowd_counting.train import *
    complete_train('Project/images')


The model can be trained via the terminal by following `the creators method <https://github.com/leeyeehoo/CSRNet-pytorch>`_


Model evaluation
*****************
The :ref:`rst_model` contain a function called evaluate that can be used to evaluate the MAE and MSE of the models.

.. code-block:: python
    
    from Crowd_counting.model import evaluate
    MAE,MSE = evaluate(my_model,img_paths)