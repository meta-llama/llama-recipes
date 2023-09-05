# Contributing to llama-recipes
We want to make contributing to this project as easy and transparent as
possible.

## Pull Requests
We actively welcome your pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Facebook's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Facebook has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## License
By contributing to llama-recipes, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.

## Tests
Llama-recipes currently comes with a basic set of unit tests but we strive to increase our test coverage in the future in order to mitigate silent errors.
When submitting a new feature PR please make sure to cover the newly added code with a unit test.
We use **pytest** for our unit tests and to run them locally you need to install llama-recipes with optional [tests] dependencies enabled:
```
pip install --extra-index-url https://download.pytorch.org/whl/test/cu118 llama-recipes[tests]
```
The unit tests can be found in the [tests](./tests/) folder and you can run them from the main directory using:
```
python -m pytest tests/
```