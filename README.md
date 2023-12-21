# Table of Content
- [Table of Content](#table-of-content)
- [GenericDecisionProcess](#genericdecisionprocess)
  - [Overview](#overview)
  - [Intended audience](#intended-audience)
  - [Install](#install)
  - [Quick Start Guide](#quick-start-guide)
  - [Tests](#tests)
  - [Examples](#examples)
  - [Contributing](#contributing)

# GenericDecisionProcess
## Overview

GenericDecisionProcess is a end-to-end library that aims to provide:

-   Intuitive formalization structures for sequential decision processes (SDPs), regardless of their components being known or black-boxed.)
-   Generic decision-making agents (e.g. reinforcement learning agents, optimization-based algorithms...) able to provide a solution to any SDP - as long as they fall into their supported sequential decision processes class.

## Intended audience
-   Researcher folks who want a quick and reliable prototype of their SDPs to being solved by a standard controller,
-   Engineer folks who want to build a scalable and reliable pipeline from SDP model to actual control,
-   Regular folks who just want to enjoy a working control on their preferred SDPs :).

## Install

Launch your favorite installer (pip, conda...) on [setup.py](https://github.com/epochstamp/decision_process/tree/main/setup.py).

## Quick Start Guide

See the following notebooks:

-   [Create a decision process](https://github.com/epochstamp/decision_process/tree/main/decision_making_process_quickie.ipynb)

## Tests

Tests have been initially written for `pytest`. The following command launches the unit tests: 

```bash
pytest tests/
```

N.B. : To launch the unit tests for the controller, you need to install `glpk` with your favorite package manager.

## Examples

Examples may be executed as simple as follows:
```bash
python examples/an_example.py
```

See the [examples](https://github.com/epochstamp/decision_process/tree/main/decision_process_examples) for more details.

## Contributing

It is important for us to keep a consistent, (almost) fully tested (with full coverage) code, even if unit tests are not perfect. This codebase follows [PEP8](https://www.python.org/dev/peps/pep-0008/) conventions. Docstrings are also put everywhere it needs to be (classes, functions, methods...).

PRs are welcome and truly encouraged, as long as they keep the abovementioned status unchanged.
