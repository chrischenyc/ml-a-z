# Coding and Learning Exercises (Machine Learning A-Z: AI, Python & R)

## Table of Contents
1. [Description](#description)
2. [Setup](#setup)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Dependencies](#dependencies)
6. [Commit Message Conventions](#commit-message-conventions)
7. [Models](#models)
   - [Simple Linear Regression](#simple-linear-regression)
8. [Fun Facts](#fun-facts)
   - [The Meaning of 42](#the-meaning-of-42)
   - [The 8 in Code Formatters](#the-8-in-code-formatters)
   - [The SciKit in scikit-learn](#the-scikit-in-scikit-learn)
   - [The Flake in flake8](#the-flake-in-flake8)

## Description

This repository contains coding exercises from the Udemy course [Machine Learning A-Z: AI, Python & R](https://www.udemy.com/course/machinelearning). The project also leverages [Cursor](https://cursor.sh/) for code generation and project structuring.

## Setup

### Prerequisites

- Python 3.x
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone git@github.com:chrischenyc/ml-a-z.git
   cd ml-a-z
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up pre-commit hooks**:
   ```bash
   pre-commit install
   ```

## Usage

To automatically run and monitor changes in Python scripts, use `auto_run.py`. This script will keep running and watch for any modifications in the `src` folder. When a change is detected, the modified file will be re-executed automatically:
   ```bash
   python auto_run.py
   ```

![Auto Run Script](docs/auto-run.png)


## Project Structure

- `src/`: Contains all source code files.
- `data/`: Contains raw training data.
- `requirements.txt`: Lists Python dependencies.
- `.gitignore`: Specifies files and directories to be ignored by Git.
- `.pre-commit-config.yaml`: Contains configuration for pre-commit hooks.
- `.flake8`: Configuration file for flake8 linting.
- `pyproject.toml`: Configuration for commit message conventions.

## Dependencies

### Core Dependencies
- [Python 3.x](https://www.python.org/)
- [pip](https://pip.pypa.io/en/stable/)

### Machine Learning & Data Science
- [NumPy](https://github.com/numpy/numpy) - Numerical computing
- [pandas](https://github.com/pandas-dev/pandas) - Data manipulation and analysis
- [scikit-learn](https://github.com/scikit-learn/scikit-learn) - Machine learning algorithms
- [matplotlib](https://github.com/matplotlib/matplotlib) - Data visualization

### Development Tools
- [pre-commit](https://github.com/pre-commit/pre-commit) - Git hook management
- [flake8](https://github.com/PyCQA/flake8) - Python code linting
- [black](https://github.com/psf/black) - Python code formatting
- [isort](https://github.com/PyCQA/isort) - Import sorting
- [watchdog](https://github.com/gorakhargosh/watchdog) - File system events monitoring
- [colorama](https://github.com/tartley/colorama) - Cross-platform colored terminal text

### Git Tools
- [commitizen](https://github.com/commitizen-tools/commitizen) - Commit message conventions
- [questionary](https://github.com/tmbo/questionary) - Interactive command line prompts

## Commit Message Conventions

This project follows the [Conventional Commits](https://www.conventionalcommits.org/) specification. For detailed information about commit types and examples, please refer to:

- [Angular Commit Message Guidelines](https://github.com/angular/angular/blob/main/CONTRIBUTING.md#-commit-message-format)
- [Conventional Commits Specification](https://www.conventionalcommits.org/en/v1.0.0/)

The commit message format is enforced using [commitizen](https://github.com/commitizen-tools/commitizen).

## Models

### Simple Linear Regression
Salary prediction based on years of experience - [src/02_regression/01_simple_linear_regression.py](src/02_regression/01_simple_linear_regression.py)

![Simple Linear Regression Training Set](output/s02_01_simple_linear_regression_training_set.png)
![Simple Linear Regression Test Set](output/s02_01_simple_linear_regression_test_set.png)

Statistical significance analysis using p-values - [src/02_regression/02_p_values.py](src/02_regression/02_p_values.py)

BTC and COIN price analysis - [src/02_regression/exec_01_BTC_COIN.py](src/02_regression/exec_01_BTC_COIN.py)

![BTC and COIN Price Analysis](output/s02_exec_01_BTC_COIN_price.png)
![BTC and COIN Price Training Set](output/s02_exec_01_BTC_COIN_price_training_set.png)
![BTC and COIN Price Test Set](output/s02_exec_01_BTC_COIN_price_test_set.png)

## Fun Facts

### The Meaning of 42
- The random state value `42` used in splitting training and test sets is a reference to ["The Hitchhiker's Guide to the Galaxy"](https://en.wikipedia.org/wiki/The_Hitchhiker%27s_Guide_to_the_Galaxy) by Douglas Adams, where 42 is the "Answer to the Ultimate Question of Life, the Universe, and Everything". This value is commonly used in machine learning as a default random seed for [reproducibility](https://scikit-learn.org/stable/common_pitfalls.html#randomness). Here's how it's used in our code:

  ```python
  from sklearn.model_selection import train_test_split

  # Split the dataset into training and test sets
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42
  )
  ```

### The SciKit in scikit-learn
- The name "scikit-learn" comes from "SciPy Toolkit" (SciKit) and its focus on machine learning. It's part of a family of SciKits built on top of [SciPy](https://www.scipy.org/), including [scikit-image](https://scikit-image.org/) for image processing and [scikit-bio](http://scikit-bio.org/) for bioinformatics.

### The Flake in flake8
- The name "flake8" combines [pyFlakes](https://github.com/PyCQA/pyflakes) (a Python code checker) and [PEP 8](https://peps.python.org/pep-0008/) (Python's style guide), showing its dual purpose of checking both code quality and style.
