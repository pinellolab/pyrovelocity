```{highlight} shell

```

# Contributing

Contributions are welcome, greatly appreciated, and credit will always be given.

You can contribute in many ways:

## Types of Contributions

### Report Bugs

Report bugs at <https://github.com/pinellolab/pyrovelocity/issues>.

If you are reporting a bug, please fill out the provided template including:

- Your operating system name and version.
- Any details about your local setup that might be helpful in troubleshooting.
- Detailed steps to reproduce the bug.

### Fix Bugs

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

### Implement Features

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

### Write Documentation

pyrovelocity could always use more documentation, whether as part of the
official pyrovelocity docs, in docstrings, or even on the web in blog posts,
articles, and such.

### Submit Feedback

The best way to send feedback is to file an issue at <https://github.com/pinellolab/pyrovelocity/issues>.

If you are proposing a feature:

- Explain in detail how it would work.
- Keep the scope as narrow as possible, to make it easier to implement.
- Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

## Get Started

### Cloud

See the [reproducibility/environment](https://github.com/pinellolab/pyrovelocity/tree/master/reproducibility/environment) folder and associated README.md.

### Local

The following is a rough guide to setting up `pyrovelocity` for local development.

1. Fork the `pyrovelocity` repo on GitHub.

2. Clone your fork locally:

   ```
   $ git clone https://github.com/your_name_here/pyrovelocity.git
   ```

3. Install your local copy with [poetry and nox](https://cookiecutter-hypermodern-python.readthedocs.io/en/2022.6.3.post1/guide.html#requirements) or with [conda]().

4. Create a branch for local development:

   ```
   $ git checkout -b name-of-your-bugfix-or-feature
   ```

   Now you can make your changes locally.

5. When you're done making changes, you can check that your changes pass the
   most basic checks implemented in noxfile.py (run `nox --list-sessions` to list all available):

   ```
   $ nox -x -rs pre-commit
   $ nox -x -rs tests-3.10
   $ nox -x -rs docs-build
   ```

   These will be confirmed via the GitHub actions workflow that will run on your fork and pull request.

6. Commit your changes and push your branch to GitHub:

   ```
   $ git add .
   $ git commit -m "Your detailed description of your changes."
   $ git push origin name-of-your-bugfix-or-feature
   ```

7. Submit a pull request through the GitHub website.

## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include pytest tests and xdoctests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring.
3. The pull request should work for Python 3.10.
