# Convert2SNN
Tool for converting conventional neural networks to spiking neural networks.

***Currently under construction. Please check again later until everything is implemented.***

## About
This tool can be used to convert and optimize conventional neural networks that were trained in TensorFlow to spiking neural networks (SNN).

## Installation
Run the following to install:
```bash
pip install convert2snn
```

## Usage
```python
from convert2snn import convert
```

## Developing
To install Convert2SNN alongside the tools you need to develop and run tests, run the following:
```bash
pip install -e.[dev]
```

## References
- [Mueller, Studenyak, Auge, Knoll, "Spiking Transformer Networks: A Rate Coded Approach for Processing Sequential Data", ICSAI, 2021](https://mediatum.ub.tum.de/1633751)
- [Mueller, Auge, Knoll, "Normalization Hyperparameter Search for Converted Spiking Neural Networks", Bernstein Conference, 2021](https://abstracts.g-node.org/conference/BC21/abstracts#/uuid/30534c50-fe09-4842-9ee6-f0127c52ce73)
- [Mueller, Hansjakob, Auge, Knoll, "Minimizing Inference Time: Optimization Methods for Converted Deep Spiking Neural Networks", IJCNN, 2021](https://ieeexplore.ieee.org/abstract/document/9533874)
- [Mueller, Hansjakob, Auge, "Faster Conversion of Analog to Spiking Neural Networks by Error Centering", Bernstein Conference, 2020](https://abstracts.g-node.org/abstracts/c4ee2b6a-340f-4955-9629-63f67ec63584)
