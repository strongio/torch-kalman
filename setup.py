from setuptools import setup, find_packages

from torchcast import __version__

setup(
    name='torchcast',
    version=__version__,
    description='Forecasting in PyTorch',
    url='http://github.com/strongio/torchcast',
    author='Jacob Dink',
    author_email='jacob.dink@strong.io',
    license='MIT',
    packages=find_packages(include='torchcast.*'),
    zip_safe=False,
    install_requires=[
        'torch>=1.7',
        'numpy>=1.4'
    ],
    tests_require=['parameterized>=0.7', 'filterpy>=1.4']
)
