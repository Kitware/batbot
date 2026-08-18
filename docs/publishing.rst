PyPI Trusted Publishing
=======================

BatBot's ``python-publish.yaml`` workflow publishes tagged releases with
OpenID Connect (OIDC). Trusted publishing exchanges GitHub's short-lived OIDC
identity for a temporary PyPI credential, so the repository does not need a
long-lived PyPI API token.

Publishing a release
--------------------

#. Update ``batbot/_version.py`` and merge the change into ``main``.
#. Wait for the test and distribution workflows to pass.
#. Create and push a matching semantic-version tag, for example:

   .. code-block:: bash

      git tag v0.3.0
      git push origin v0.3.0

#. Approve the ``pypi`` GitHub environment deployment if protection rules
   require it.
#. Confirm that the ``Publish to PyPI`` job completed and verify the files on
   `PyPI's BatBot page <https://pypi.org/project/batbot/>`_.

The publish job needs ``permissions: id-token: write`` and the
``pypa/gh-action-pypi-publish`` step must omit ``password``. Those settings are
already present in this repository's workflow.
