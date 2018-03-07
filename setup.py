from setuptools import setup

setup(name='kalman_pytorch',
      version='0.1',
      description='Kalman Filter using pytorch for parameter-estimation',
      url='http://github.com/jwdink/kalman_pytorch',
      author='Jacob Dink',
      author_email='jacobwdink@gmail.com',
      license='MIT',
      packages=['kalman_pytorch'],
      zip_safe=False,
      install_requires=['torch', 'numpy'])
