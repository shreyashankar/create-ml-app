# create-ml-app

`create-ml-app` makes it easier to spin up a machine learning project locally in Python and handle various package dependencies. The name is inspired by [`create-react-app`](https://github.com/facebook/create-react-app). 

This project abstracts away `pip install`s and virtual environment commands from the user. To use, simply fork this repository and execute `make run` from your shell. The `main.py` file in this repo has an example ML script (training a neural network on MNIST in PyTorch).

## Motivation

When starting a new ML project or prototyping a model locally, it can be tedious to:

* handle all the Python package dependencies
* create/activate/deactivate your virtual environment
* parameterize arguments
* remember to define a random seed

Having a Makefile can simplify the virtual environment overhead and centralize parameters in one place. This repository is an example of how to use a Makefile in a simple ML project (training a neural network on MNIST in PyTorch). 

## Background

Under the hood, this project uses `venv` to create a virtual environment and install Python packages. The primary commands supported by this Makefile are:

* `make lint`: This will show errors as flagged by pylint.
* `make run`: This will download any new packages found in `setup.py` and run `main.py` with user-specified variables. You may need to modify the Makefile to include variables of your choice and change the `run` definition to run your Python file with your specified variables.

If you want to use any Python package in your project, simply add the package name to `setup.py` and it will get installed the next time you execute `make run` from your shell.

## Usage

    git clone https://github.com/shreyashankar/create-ml-app.git my-ml-app
    cd my-ml-app
    make run