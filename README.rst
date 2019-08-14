pyroc
=========

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

Documentation
-------------

Documentation is available on `readthedocs <http://pyroc.readthedocs.io/en/latest/>`_. An executable demonstration of the package is available on `GitHub as a Jupyter Notebook <https://github.com/alistairewj/pyroc/blob/master/usage.ipynb>`_.

Installation
------------

To install the package with pip, run::

    pip install pyroc

To install this package with conda, run::
    
    conda install -c conda-forge pyroc