import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='convert2snn',
    version='0.1.0',
    author='Etienne Mueller',
    author_email='contact@etiennemueller.de',
    description='Convert conventional to spiking neural networks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/EtienneMueller/Convert2SNN',
    project_urls={
        "Bug Tracker": "https://github.com/EtienneMueller/Convert2SNN/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    keywords='spiking neural networks, tensorflow, deep learning, AI',
    packages=setuptools.find_packages(where="src"),
    package_dir={'': 'src'},
    python_requires=">=3.10",
    install_requires=[
        "tensorflow==2.11.0",
        "numpy==1.24"
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
            "check-manifest>=0.47",
            "twine>=4.0"
        ],
    },
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'src=src.cli:main',
        ],
    },
)
