pyroc
=========

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6819206.svg
   :target: https://doi.org/10.5281/zenodo.6819206

pyroc is a package for analyzing receiver operator characteristic (ROC) curves.
It includes the ability to statistically compare the area under the ROC (AUROC) for two or more classifiers.

Quick start
-----------

Install:

    pip install pyroc

Use:

    import pyroc
    import numpy as np
    
    pred = np.random.rand(100) 
    target = np.round(pred)
    # flip 10% of labels
    target[0:10] = 1 - target[0:10]
    W = pyroc.auroc(target, pred)

    # second prediction
    pred2 = pred
    pred2[10:20] = 1 - pred2[10:20]
    auroc, ci = pyroc.auroc_ci(target, [pred, pred2])
    print(auroc)
    print(ci)

A usage.ipynb notebook is provided demonstrating common usage of the package (requires Jupyter: `pip install jupyter`).

Documentation
-------------

Documentation is available on `readthedocs <http://pyroc.readthedocs.io/en/latest/>`_. An executable demonstration of the package is available on `GitHub as a Jupyter Notebook <https://github.com/alistairewj/pyroc/blob/master/usage.ipynb>`_.

Installation
------------

To install the package with pip, run::

    pip install pyroc

To install this package with conda, run::
    
    conda install -c conda-forge pyroc

Acknowledgement
---------------

Please use the latest DOI on `Zenodo`_. Example BibTeX:

.. code-block:: latex

    @software{pyroc,
      author       = {Alistair Johnson and
                      Lucas Bulgarelli and
                      Tom Pollard},
      title        = {alistairewj/pyroc: pyroc v0.2.0},
      month        = jul,
      year         = 2022,
      publisher    = {Zenodo},
      version      = {v0.2.0},
      doi          = {10.5281/zenodo.6819206},
      url          = {https://doi.org/10.5281/zenodo.6819206}
    }


.. _Zenodo: https://doi.org/10.5281/zenodo.6819205
