PyPI Trusted Publishing
=======================

BatBot's ``python-publish.yaml`` workflow publishes tagged releases with
OpenID Connect (OIDC). Trusted publishing exchanges GitHub's short-lived OIDC
identity for a temporary PyPI credential, so the repository does not need a
long-lived PyPI API token.

One-time PyPI configuration
---------------------------

#. Sign in to `PyPI <https://pypi.org/>`_ with an account that owns the
   ``batbot`` project.
#. Open ``Manage project`` for ``batbot``, choose ``Publishing``, and add a
   GitHub trusted publisher.
#. Enter these values exactly:

   * Owner: ``Kitware``
   * Repository: ``batbot``
   * Workflow: ``python-publish.yaml``
   * Environment: ``pypi``

#. Save the publisher. For a package that has not been uploaded yet, create a
   pending publisher with the same values under the PyPI account's publishing
   settings.

One-time GitHub configuration
-----------------------------

#. Open the repository's ``Settings`` page, then ``Environments``.
#. Create an environment named ``pypi``. The name must match both PyPI and the
   workflow.
#. Add required reviewers and a deployment tag rule such as ``v*`` if release
   approval is desired.
#. Remove the obsolete ``BATBOT_PYPI_TOKEN`` Actions secret after one trusted
   release succeeds. The workflow must not pass a username or password to the
   PyPI publish action.

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
