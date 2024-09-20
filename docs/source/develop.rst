Development
===========

Please refer to the style and contribution guidelines documented in the
`IRF Software Contribution Guide <https://danielk.developer.irf.se/software_contribution_guide/>`_.
Generally external code-contributions are made trough a "Fork-and-pull"
workflow, while internal contributions follow the branching strategy outlined
in the contribution guide.

Please refer to `GitHub
<https://github.com/danielk333/hardtarget/blob/main/DEVELOP.md/>`_ for more
details.




Documentation
"""""""""""""

Hardtarget uses google style for inline comments.

In order to build html or apidoc, make sure virtual environment is pip installed with *develop* flag.

To generate apidoc and html, or clean build stuff

.. code-block:: bash

    cd hardtarget

    make -C docs/ apidoc
    make -C docs/ html
    make -C docs/ clean
    make -C docs/ realclean




