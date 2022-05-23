.. _rst_install:


Installation
============

First make sure that you have git-lfs installed and correcly configured. This is used to download the pretrain models weights that are heavier than 100MB during the library installation.
This step is not needed if you don't want to load already trained models.

.. code-block:: bash

   $ git lfs install --skip-repo
   
   
Then install the package using pip 

.. code-block:: bash
    
    $ pip install git+https://github.com/lrobins1/crowd_counting.git
   