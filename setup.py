from setuptools import setup

setup(name='torch_kalman',
      version='0.1',
      description='Kalman Filter using pytorch for parameter-estimation',
      url='http://github.com/strongio/torch_kalman',
      author='Jacob Dink',
      author_email='jacob.dink@strong.io',
      license='MIT',
      packages=['torch_kalman'],
      zip_safe=False,
      install_requires=['torch', 'numpy'])
