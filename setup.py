from setuptools import setup, find_packages

setup(name='torch_kalman',
      version='0.1',
      description='Kalman filters with pytorch',
      url='http://github.com/strongio/torch_kalman',
      author='Jacob Dink',
      author_email='jacob.dink@strong.io',
      license='MIT',
      packages=[p for p in find_packages() if 'torch_kalman' in p],
      zip_safe=False,
      install_requires=['torch>=1.0.0', 'numpy>=1.14.2', 'tqdm'],
      test_suite='nose.collector',
      tests_require=['nose'])
