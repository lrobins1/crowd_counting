Load model and make predictions
================================

Available pretrained Models
****************************

Three pretrained models are available to be loaded directly from the librairy : 

- shanghaiA : This model is trained on the Shanghai dataset part A. It is best suited for really crowded scenes.

- shanghaiB : This model is trained on the Shanhai dataset part B. It is best suited for middle crowded scenes.

- A10 : This model is a model trained on the A10 auditorium in Louvain-la-neuve. It is best suited to count the number of people in any auditorium.

Loading a model
*****************

The :ref:`rst_model` contain two functions to load models : 

The first one is used to load one of the pretrained models available in the librairy. It can be used as follows.

.. code-block:: python
    
    from Crowd_counting.model import *
    my_model = load_pretrained('shangaiA')
    
The second one is used to load a model from a .tar archive as produced while training your own models.

.. code-block:: python
    
    from Crowd_counting.model import *
    my_model = load_model('path/to/archie.tar')

Make predictions
*****************

The :ref:`rst_model` contain one function called predict that take an image and predict two things : the number of people and a density map. It can be used as in the following exemple :

.. code-block:: python
    
    from Crowd_counting.model import *
    people_number, density_map = predict(my_model, 'path/to/image.png')
    
The density map can then be seen using the visualization function

.. code-block:: python
    
    from Crowd_counting.model import *
    visualize("path/to/image.png", model = my_model)