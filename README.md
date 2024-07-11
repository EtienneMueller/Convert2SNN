# Convert2SNN
**Convert2SNN** is a flexible tool for converting conventional TensorFlow-trained artificial neural networks (ANNs) into spiking neural networks (SNNs). This tool helps researchers and developers integrate SNNs into their workflows, making it easier to create energy-efficient models suited for neuromorphic hardware.

Spiking neural networks can potentially achieve high energy efficiency, especially when combined with sparsely activated layers. Convert2SNN offers options to convert models into different types of spiking neurons, including rate-coded, population-coded, and temporally-coded neurons.

> **Note:** *This tool is currently further being developed to add features based on my ongoing research at the University of Melbourne in Computational Neuroscience. If I broke something relevant to you don't hesitate raising an issue.*

---

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Example](#example)
- [Developing](#developing)
- [References](#references)
- [License](#license)
- [Contributing](#contributing)

---

## Features
- **Flexible Conversion Options**: Convert your neural network to rate-coded, population-coded, or temporally-coded SNNs.
- **Compatibility with TensorFlow**: Directly convert TensorFlow-trained models without retraining.
- **Optimized for Energy Efficiency**: Designed with neuromorphic computing in mind, this tool optimizes model configurations to improve energy efficiency.
- **Minimal Accuracy Loss**: Converts networks while preserving accuracy as closely as possible.
- **Batch Conversion**: Supports running multiple conversions for comparison of energy efficiency and performance metrics.

---

## Installation
To install Convert2SNN, install this repository via:

```bash
pip install git+https://github.com/EtienneMueller/Convert2SNN.git
```

Alternatively, it is available on [PyPI](https://pypi.org/project/convert2snn/) (make sure it is the latest version):

```bash
pip install convert2snn
```

If you want to use the latest development version, clone the repository and install in editable mode:

```bash
git clone https://github.com/EtienneMueller/Convert2SNN.git
cd Convert2SNN
pip install -e .[dev]
```

## Usage

Convert a trained TensorFlow model to an SNN:

```python
from convert2snn import convert

# Load your TensorFlow model
model = load_model('path/to/model')

# Convert to a spiking neural network (choose from 'rate', 'population', or 'temporal')
snn_model = convert(model, conversion_type='rate')
```

## Example

An example script demonstrating the training of a simple neural network, its conversion to a spiking model, and evaluation of its energy efficiency can be found in the examples/ directory:

```
python examples/train_example.py
```

This script provides a hands-on example to quickly understand how Convert2SNN works.
Developing

To set up a development environment with tools for testing and development, use:

```
pip install -e .[dev]
```

## Running Tests

To ensure that all components are functioning as expected, run the tests:

```bash
pytest
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature-branch).
3. Make your changes and add tests if necessary.
4. Open a pull request.

For major changes, please open an issue to discuss the proposed updates before starting work.

## References

This tool builds on research presented in the following publications:

### Spike Encoding Techniques

Auge, Hille, Mueller, Knoll, ["A survey of encoding techniques for signal processing in spiking neural networks"](https://link.springer.com/article/10.1007/s11063-021-10562-2), Neural Processing Letters, 2021

### Rate Coded Conversion

Mueller, Hansjakob, Auge, Knoll, ["Minimizing Inference Time: Optimization Methods for Converted Deep Spiking Neural Networks"](https://ieeexplore.ieee.org/abstract/document/9533874/), IJCNN, 2021

Mueller, Studenyak, Auge, Knoll, ["Spiking Transformer Networks: A Rate Coded Approach for Processing Sequential Data"](https://ieeexplore.ieee.org/abstract/document/9664146), ICSAI, 2021
    
Mueller, Auge, Knoll, ["Normalization Hyperparameter Search for Converted Spiking Neural Networks"](https://abstracts.g-node.org/conference/BC21/abstracts#/uuid/30534c50-fe09-4842-9ee6-f0127c52ce73), Bernstein Conference, 2021

Mueller, Hansjakob, Auge, ["Faster Conversion of Analog to Spiking Neural Networks by Error Centering"](https://abstracts.g-node.org/conference/BC20/abstracts#/uuid/c4ee2b6a-340f-4955-9629-63f67ec63584), Bernstein Conference, 2020

### Population Coded Conversion

Mueller, Auge, Knoll, ["Exploiting Inhomogeneities of Subthreshold Transistors as Populations of Spiking Neurons"](https://link.springer.com/chapter/10.1007/978-3-031-20738-9_55), ICNC-FSKD, 2022

### Temporal Conversion

Mueller, Auge, Klimaschka, Knoll, ["Neural Oscillations for Energy-Efficient Hardware Implementation of Sparsely Activated Deep Spiking Neural Networks"](https://practical-dl.github.io/2022/long_paper/16.pdf), AAAI Practical DL, 2022 



## License

Convert2SNN is licensed under the MIT License. See the LICENSE file for more details.
