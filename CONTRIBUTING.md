# Contributing to MtReNIRS

Thank you for your interest in contributing to MtReNIRS. Bug reports,
documentation improvements, tests, and focused code changes are welcome.

## Before you start

- Search existing issues and pull requests before opening a new one.
- Open an issue before making a substantial behavioral or API change so the
  approach can be discussed first.
- Report suspected vulnerabilities privately as described in
  [SECURITY.md](SECURITY.md), not through a public issue.
- Do not contribute data, model artifacts, or sensor geometries unless you have
  the right to redistribute them under terms compatible with this project.

## Development setup

MtReNIRS contains two independently installable Python packages. From the
repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "./alignment[test]" -e "./models[test,raw,muse]"
```

Run the package test suites before submitting a change:

```bash
python -m pytest alignment/tests models/tests
```

Keep changes focused, add or update tests for changed behavior, and update the
relevant documentation when public APIs or user workflows change.

## Pull requests

A pull request should explain the problem, the chosen approach, and how the
change was tested. Link related issues and keep unrelated changes in separate
pull requests.

Most contributions require you to agree to a Contributor License Agreement
(CLA) declaring that you have the right to, and do, grant us the rights to use
your contribution. For details, visit
[Microsoft CLA](https://cla.opensource.microsoft.com/). When you submit a pull
request, a CLA bot will automatically determine whether you need to provide a
CLA and annotate the pull request appropriately. Follow the bot's instructions
if action is required.

## Code of conduct

This project follows the [Microsoft Open Source Code of Conduct](CODE_OF_CONDUCT.md).
By participating, you are expected to uphold it.