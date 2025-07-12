from setuptools import setup, find_packages

setup(
    name='linear_regression',
    version='0.1.0',
    description='Linear Regression Implementation from Scratch',
    author='Your Name',
    author_email='your.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
        'matplotlib>=3.5.0; platform_system != "Windows"',
        'pytest>=7.0.0; extra=="test"'
    ],
    extras_require={
        'test': ['pytest>=7.0.0']
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
)
