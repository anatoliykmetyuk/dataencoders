from setuptools import setup
from setuptools import find_packages

version = '0.1.0'

setup(name='Data Encoders',
      version=version,
      description='Tools to quickly encode and decode data for machine learning',
      author='Anatolii Kmetiuk',
      author_email='anatoliykmetyuk@gmail.com',
      url='https://github.com/anatoliykmetyuk/dataencoders',
      license='MIT',
      install_requires=['numpy'],
      packages=find_packages())