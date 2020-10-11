# GBS-matrix-similarity
Xanadu's [StrawberryFields](https://github.com/XanaduAI/strawberryfields) provides a functionality to use Gaussian Boson Sampling (GBS) to measure similarity between graphs [1].

This repository provides a bit different implementation of the similarity functionality in StrawberryFields to provide a few
1. GBS-matrix-similarity uses `thewalrus` library directly and avoids the overhead of embedding a matrix into a GBS device.
2. GBS-matrix-similarity allows the embedding of arbitrary symmetric matrices, not only adjacency matrices of non-weighted graphs.
3. GBS-matrix-similarity optionally allows embedding of diagonal terms into the vector of means.

The core of the package is the `GBSDevice` class, which is a simple wrapper allowing the embedding of matrices into GBS and sampling. As an example use case, the package also provides some additional files for extracting symmetric matrices from second quantized molecular hamiltonians, encoding them into GBS with the intent to measure some kind of similarity between molecules.

## References
[1] Thomas R. Bromley, et. al. "Applications of Near-Term Photonic Quantum Computers: Software and Algorithms", [Quantum Sci. Technol. 5 034010 (2020)](https://iopscience.iop.org/article/10.1088/2058-9565/ab8504/meta).
