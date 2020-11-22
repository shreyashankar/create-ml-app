from setuptools import setup, find_packages

setup(
    name='my-app',
    version='0.1',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'pylint',
        'torch',
        'torchvision'
    ]
)