from setuptools import setup

setup(name='torch-kalman',
      version='0.1',
      description='Kalman Filter using pytorch for parameter-estimation',
      url='http://github.com/strongio/torch-kalman',
      author='Jacob Dink',
      author_email='jacob.dink@strong.io',
      license='MIT',
      packages=['torch-kalman'],
      zip_safe=False,
      install_requires=['torch', 'numpy'])
