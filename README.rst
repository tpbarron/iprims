===============================
Interaction Primitives
===============================


.. image:: https://img.shields.io/pypi/v/iprims.svg
        :target: https://pypi.python.org/pypi/iprims

.. image:: https://img.shields.io/travis/tpbarron/iprims.svg
        :target: https://travis-ci.org/tpbarron/iprims

.. image:: https://readthedocs.org/projects/iprims/badge/?version=latest
        :target: https://iprims.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

.. image:: https://pyup.io/repos/github/tpbarron/iprims/shield.svg
     :target: https://pyup.io/repos/github/tpbarron/iprims/
     :alt: Updates


A basic implementation of the Interaction Primitives framework for human robot collaboration


* Free software: MIT license
* Documentation: https://iprims.readthedocs.io.


Features
--------

This is a (mostly complete) implementation of the Interaction Primitives
framework described Dr. Heni Ben Amor and others at TU Darmstadt.

See these papers for reference:

Ewerton, M.; Neumann, G.; Lioutikov, R.; Ben Amor, H.; Peters, J.; Maeda, G. (2015).
    Learning Multiple Collaborative Tasks with a Mixture of Interaction Primitives,
    Proceedings of 2015 IEEE International Conference on Robotics and Automation (ICRA).

Ben Amor, H.; Neumann, G.; Kamthe, S.; Kroemer, O.; Peters, J. (2014).
    Interaction Primitives for Human-Robot Cooperation Tasks ,
    Proceedings of 2014 IEEE International Conference on Robotics and Automation (ICRA).

Maeda, G.J.; Ewerton, M.; Lioutikov, R.; Ben Amor, H.; Peters, J.; Neumann, G. (2014).
    Learning Interaction for Collaborative Tasks with Probabilistic Movement Primitives,
    Proceedings of the International Conference on Humanoid Robots (HUMANOIDS)

Probabilistic movement primitives. A Paraschos, C Daniel, JR Peters, G Neumann.
    Advances in neural information processing systems, 2013.


- TODO:
-- What if the trajectories are longer than 100 timesteps? Is that a problem? Do I just interpolate to 100 timesteps?
-- Or do I consider the trajectories to have more timesteps? How do I deal with trajectories of different lengths?
-- Run Baxter IP either keeping obs trajectories random or by expanding code to deal with no human or controlled agent
--


Credits
---------

This package was created with Cookiecutter_ and the `audreyr/cookiecutter-pypackage`_ project template.

.. _Cookiecutter: https://github.com/audreyr/cookiecutter
.. _`audreyr/cookiecutter-pypackage`: https://github.com/audreyr/cookiecutter-pypackage

