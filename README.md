# MoFlow

MoFlow is a deep learning framework for **multi-omic RNA velocity modeling** that extends the relay velocity model (cellDancer) by incorporating chromatin accessibility.  
By leveraging gene- and cell-specific kinetic parameters, MoFlow can jointly model chromatin accessibility, transcription, splicing, and degradation, enabling the study of transcriptional dynamics across diverse cell states.

---

## Installation

Clone the repository and set up a conda environment:

```bash
git clone https://github.com/AriHong/MoFlow.git
cd MoFlow

conda create -n moflow python=3.7.0
conda activate moflow
pip install -r requirements.txt
```

---

## Quick Start

We provide a demonstration notebook under `notebooks/Demo.ipynb` showing how to run MoFlow on a toy dataset and compute downstream scores.
Also, the repository includes notebooks under `notebooks/` for reproducing figures from the manuscript.

---

## Acknowledgements

Portions of this code are adapted from the **cellDancer** repository:  
[https://github.com/GuangyuWangLab2021/cellDancer/](https://github.com/GuangyuWangLab2021/cellDancer/)

We thank the authors of cellDancer and MultiVelo for making their work publicly available.



