# Memory Layers with Sparse Memory Finetuning

Re-implementation of memory layer and sparse memory finetuning from Lin et. al (2025). See Arxiv: https://arxiv.org/abs/2510.15103

The memory layer includes:

- [x] Product Key Optimization from Lample et al. (2019) (see Arxiv: https://arxiv.org/abs/1907.05242):   
    - [x] Developed custom `nn.EmbeddingBag` implementation from research paper description.
    - [ ] Developed Cuda version using Triton Language.
    - [ ] Develop Pallas version for TPU and SparseCore utilisation.

- [x] Memory+ architecture from Berges et al. (2024) (see Arxiv: https://arxiv.org/abs/2412.09764)

- [ ] Sparse Memory Finetuning from Lin et al. (2025) (see Arxiv: https://arxiv.org/abs/2510.15103)