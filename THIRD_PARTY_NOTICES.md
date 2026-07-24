# Third-party validation data

The Apache-2.0 license for GBSKernels source code does not relicense the
third-party scientific data under `examples/jiuzhang/validation_data/`.

The files `click_probs_squeezed_0.npy` and
`click_probs_squashed_0.npy` are retained Jiuzhang 1.0 probability arrays from:

- Javier Martinez-Cifuentes, Karen Milena Fonseca-Romero, and Nicolas Quesada,
  "Classical models are a better explanation of the Jiuzhang 1.0 Gaussian
  Boson Sampler than its targeted squeezed light model," Zenodo record
  [7194775](https://doi.org/10.5281/zenodo.7194775), licensed
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

The files `T_full.npy`, `events_band13_32.npy`,
`squeezing parameters.txt`, and `empirical_click_rates.npy` are small derived
validation inputs from the Jiuzhang 1.0 data published by Han-Sen Zhong et al.:

- [Experimental raw data of "Quantum computational advantage using
  photons"](https://quantum.ustc.edu.cn/web/en/node/915), University of Science
  and Technology of China, 2020.

The USTC page makes the raw data publicly downloadable but does not state a
separate data license. These derived inputs are included for validation,
attribution, and reproducibility; no Apache-2.0 license is asserted over the
underlying USTC data. Every redistributed or derived file is bound to the
SHA-256 digest in `scripts/prepare_validation_data.py`.
