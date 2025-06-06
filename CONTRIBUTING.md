# Contributing to Optimizers
We want to make contributing to this project as easy and transparent as
possible. Our goal is to provide a repo that promotes optimizer research
and development separate from the official PyTorch library. Please only
create pull requests for improving existing optimizers in the repo; new
optimizers should be created in a separate public repo.

## Pull Requests
We actively welcome your pull requests for existing optimizers.

1. Fork the repo and create your branch from `main`. Install the package inside of your Python environment with `pip install -e ".[dev]"`. Run `pre-commit install` to set up the git hook scripts.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes. To run the subset of the tests that can be run on CPU use `make test`; to run the tests for a single GPU use `make test-gpu` and to run the subset of tests that require 2-4 GPUs use `make test-multi-gpu`.
5. Make sure your code passed pre-commit. You can use `pre-commit run --all-files` to automatically lint and format the code where possible. Use `make type-check` for type checking.
6. If you haven't already, complete the Contributor License Agreement ("CLA").

> [!NOTE]
> If you add a new class or function that a user of the package might want to interact with directly, make sure to add it [here](distributed_shampoo/__init__.py).

## Contributor License Agreement ("CLA")
In order to accept your pull request, we need you to submit a CLA. You only need
to do this once to work on any of Meta's open source projects.

Complete your CLA here: <https://code.facebook.com/cla>

## Issues
We use GitHub issues to track public bugs. Please ensure your description is
clear and has sufficient instructions to be able to reproduce the issue.

Meta has a [bounty program](https://www.facebook.com/whitehat/) for the safe
disclosure of security bugs. In those cases, please go through the process
outlined on that page and do not file a public issue.

## Coding Style
* Use [`ruff`](https://docs.astral.sh/ruff/) for linting and formatting
* Please maintain a consistent style with the rest of the code

## License
By contributing to Optimizers, you agree that your contributions will be licensed
under the LICENSE file in the root directory of this source tree.
