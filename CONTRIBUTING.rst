============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every
little bit helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/spotify/confidence/issues

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug"
and "help wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

Confidence could always use more documentation, whether as part of the
official Confidence docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/spotify/confidence/.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `confidence` for local development.

**Prerequisites:**

* `uv <https://docs.astral.sh/uv/>`_ - Fast Python package installer (recommended)
* Python 3.9 or later

1. Fork the `confidence` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_username/confidence.git
    $ cd confidence

3. Set up your development environment using uv::

    $ uv venv
    $ uv pip install -e ".[dev]"

   This creates a virtual environment and installs the package in editable mode with all development dependencies.

4. Verify your setup by running the tests::

    $ uv run pytest

   This should run all tests and show they pass.

5. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

6. When you're done making changes, check that your changes pass all quality checks::

    $ uv run black spotify_confidence tests --line-length 119  # Format code
    $ uv run flake8 spotify_confidence tests                   # Lint code
    $ uv run pytest                                            # Run tests

   To test across all supported Python versions (3.9, 3.10, 3.11, 3.12)::

    $ uv run tox -p auto

   Note: tox requires all Python versions to be installed on your system.

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.md.
3. The pull request should work for Python 3.9, 3.10, 3.11, and 3.12. The CI
   pipeline will automatically test all supported Python versions.

Tips
----

To run a subset of tests::

    $ uv run pytest tests/frequentist/test_ttest.py

To run a specific test::

    $ uv run pytest tests/frequentist/test_ttest.py::TestCategorical::test_summary

To run tests with verbose output::

    $ uv run pytest -v

To see test coverage::

    $ uv run pytest --cov=spotify_confidence --cov-report=html
    $ open htmlcov/index.html


Release Process
-----------------------

While commits and pull requests are welcome from  any contributor, we try to
simplify the distribution process for everyone by managing the release
process with specific contributors serving in the role of Release Managers.

Release Managers are responsible for:

* Finding a proper reviewer for each Pull Request
* Deciding what changes constitute a new Release
* Making a new Release available on Artefactory/internal PyPi

The current Release Managers are:

* Per Sillrén

Versioning
~~~~~~~~~~~
Releases follow the `Semantic Versioning standard <https://semver.org/>`_.

Given a version number MAJOR.MINOR.PATCH, increment the:

MAJOR version when you make incompatible API changes,
MINOR version when you add functionality in a backwards-compatible manner, and
PATCH version when you make backwards-compatible bug fixes.

Release Stategy
~~~~~~~~~~~~~~~~
Each new release will be made on its own branch, with the branch Master
representing the most recent, furthest release. Releases are published to PyPi
automatically once a new release branch is merged to Master. Additionally,
rew releases are also tracked manually on `github
<https://github.com/spotify/confidence/releases>`_.
