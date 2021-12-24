import setuptools


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='convert2snn',
    version='0.0.1',
    author='Etienne Mueller',
    author_email='etienne.mueller@tum.de',
    description='Convert convention to spiking neural networks.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/EtienneMueller/Convert2SNN',
    project_urls={
        "Bug Tracker": "https://github.com/EtienneMueller/Convert2SNN/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow>=2.7.0"
    ],
    extras_require={
        "dev": [
            "pytest>=3.7",
            "check-manifest>=0.47"
        ],
    },
)
