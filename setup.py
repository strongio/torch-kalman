from setuptools import setup

setup(name='kalman_pytorch',
      version='0.1',
      description='Kalman Filter using pytorch for parameter-estimation',
      url='http://github.com/strongio/kalman_pytorch',
      author='Jacob Dink',
      author_email='jacob.dink@strong.io',
      license='MIT',
      packages=['kalman_pytorch'],
      zip_safe=False,
      install_requires=['torch', 'numpy'])
