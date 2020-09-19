from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

# NOTE THIS SETUP FILE DOES NOT WORK YET

extensions = [
  Extension(
      name='im2col_cython',
      sources=['nn/im2col_cython.pyx'],
      include_dirs=['nn/'],
      language='c',
  ),
]


setup(
    name='nn',
    version='1.0.0',
    python_requires='>=3.8.0',
    packages=find_packages(),
    requires=['numpy', 'matplotlib'],
    setup_requires=['cython'],
    ext_modules=cythonize(extensions),
)
