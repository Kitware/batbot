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

Manual validation or release recovery
-------------------------------------

The Python distributions, Testing, and Docker workflows all support
``workflow_dispatch`` from GitHub's **Actions** tab. Running Testing or Docker
manually validates the selected ref; it does not publish an image. A manual
Python distributions run also defaults to validation only.

To publish an existing tag whose original push did not start a release:

#. Merge the workflow changes into ``main`` first.
#. Open **Actions → Python distributions → Run workflow**.
#. Select **main** as the workflow ref. This uses current release tooling even
   if the old tag does not contain a manual trigger or release helper.
#. Enter the existing tag, for example ``v0.2.0``, in **release_tag**.
#. Leave **publish** unchecked to test the release first. To publish it, start
   a new run with the same tag and **publish** checked.
#. Approve any required ``pypi`` environment deployment and verify PyPI.

The equivalent explicit publishing request with the GitHub CLI is:

.. code-block:: bash

   gh workflow run python-publish.yaml --repo Kitware/batbot --ref main \
     -f release_tag=v0.2.0 -F publish=true

Release preparation requires a real ``refs/tags/`` reference, checks the tag
against the version in that commit's ``batbot/_version.py`` without importing
the package, and resolves it to an immutable commit. The build and all 12
installed-wheel checks use that same commit. Publication is permitted only
after those checks pass and either a version tag was pushed or an explicit
manual publishing request selected a release tag. Branch and pull-request runs
cannot publish. Never delete or move a release tag just to retry a workflow.

The trusted publisher still uses the same workflow filename and ``pypi``
environment. Existing environment protection rules remain enforced. A manual
run started from ``main`` is subject to rules for that workflow ref, even when
it builds a tag; if the environment allows only tag-triggered deployments, a
maintainer must review the intended recovery policy before using this route.
This workflow does not bypass approvals or change environment permissions.

PyPI does not allow replacing an already uploaded distribution with the same
filename. Do not use recovery to overwrite an existing release; publish a new
version if its contents need to change.

Workflow conventions and troubleshooting
----------------------------------------

All workflow files use the same ordering: name, triggers, default read-only
permissions, concurrency, Bash run defaults, optional shared environment, and
jobs. Steps have descriptive names, checkouts do not persist credentials, jobs
have explicit timeouts, and Python dependency caches name ``pyproject.toml``.
Only publishing jobs receive additional write permissions. Superseded
pull-request runs can be cancelled; an in-progress release is not cancelled
when another push arrives. Testing continues to run on all pushed branches,
while distribution and Docker publishing keep their ``main``/version-tag
triggers. CodeQL has been removed from the repository's workflows.

If a tag exists but no Python distributions run appears, first check the
workflow's enabled state in Actions. A tag does not prove that a release job
ran, and creating a GitHub Release is not required by these push triggers.
After resolving the trigger problem, use the manual recovery route above.
If a run exists, inspect release preparation, wheel validation, environment
approval, and trusted-publisher errors in that order. Dispatching a workflow
does not guarantee publication; every release gate must still succeed.
