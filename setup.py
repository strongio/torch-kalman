from setuptools import setup, find_packages

from torch_kalman import __version__

setup(
    name='torch_kalman',
    version=__version__,
    description='Kalman filters with pytorch',
    url='http://github.com/strongio/torch_kalman',
    author='Jacob Dink',
    author_email='jacob.dink@strong.io',
    license='MIT',
    packages=find_packages(include='torch_kalman.*'),
    zip_safe=False,
    install_requires=[
        'torch>=1.1',
        'numpy>=1.4',
        'tqdm>=4.0',
        'filterpy>=1.4',
        'lazy_object_proxy>=1.4'
    ],
    test_suite='nose.collector',
    tests_require=['nose']
)
