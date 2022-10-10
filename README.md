# Slap4slip

This repository contains the code for the NAACL 2022 Findings paper 
[Modeling Ideological Salience and Framing in Polarized Online Groups with Graph Neural Networks and Structured Sparsity]
(https://aclanthology.org/2022.findings-naacl.41.pdf).

# Dependencies

The code requires `Python>=3.6`, `numpy>=1.18`, `torch>=1.2`, and `torch_geometric>=1.6`.

# Usage

To replicate the experiments using SF-SGAE, S-SGAE, and F-SGAE, run the script `src/train.sh`.
To replicate the experiment using SF-SLAE, run the script `src/train_linear.sh`.
To replicate the experiment using SF-GAE, run the script `src/train_nonsparse.sh`.

# Citation

If you use the code in this repository, please cite the following paper:

```
@inproceedings{hofmann2022slap4slip,
    title = {Modeling Ideological Salience and Framing in Polarized Online Groups with Graph Neural Networks and Structured Sparsity},
    author = {Hofmann, Valentin and Dong, Xiaowen and Pierrehumbert, Janet and Sch{\"u}tze, Hinrich},
    booktitle = {Findings of the Association for Computational Linguistics: NAACL 2022},
    year = {2022}
}
```