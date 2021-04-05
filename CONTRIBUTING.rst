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

1. Fork the `confidence` repo on GitHub.
2. Clone your fork locally::

    $ git clone https://github.com/spotify/confidence

3. Install your local copy into a virtualenv. Assuming you have virtualenvwrapper installed, this is how you set up your fork for local development::

    $ mkvirtualenv confidence_dev
    $ cd confidence/
    $ tox

   The tox command will install the dev requirements in requirements_dev.txt and run all tests.

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, check that your changes pass flake8 and the tests, including testing other Python versions with tox::

    $ flake8 confidence tests
    $ python setup.py test or py.test
    $ tox

   To get flake8 and tox, just pip install them into your virtualenv.

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for Python 3.6 and 3.7. Check
   and make sure that the tests pass for all supported Python versions.

Tips
----

To run a subset of tests::

$ py.test tests.test_confidence


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
