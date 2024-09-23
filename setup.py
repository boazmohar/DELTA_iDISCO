from setuptools import setup, find_packages

setup(
    name='DELTA_iDISCO',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'itk',
        'itk-elastix',
        'numpy',
        'pandas',
        'scikit-image',
        'h5py',
    ],
    entry_points={
        'console_scripts': [
            'delta_idisco=src.main:main',
        ],
    },
)
