# Coding and Learning Exercises (Machine Learning A-Z: AI, Python & R)

## Table of Contents
1. [Description](#description)
2. [Setup](#setup)
   - [Prerequisites](#prerequisites)
   - [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)

## Description

This repository contains coding exercises from the Udemy course [Machine Learning A-Z: AI, Python & R](https://www.udemy.com/course/machinelearning). The project also leverages Cursor for code generation and project structuring.

## Setup

### Prerequisites

- Python 3.x
- pip

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
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
