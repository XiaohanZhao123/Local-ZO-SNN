# Code for SparseZO

## Refractory of SparseZO

The new code is written with Pytorch's built-in torch.sparse library with more concise and the autograd.Function class is warped and managed by torch.nn class, which is more convenient to use.

### Current Implementation of Sparsity

Currently, the sparsity during backprop is implemented by the following two ways, and we want to make the
API with more flexibility, making them open to change:

* No-restoring mode: just use torch.sparse all the way when necessary. However, this mode ask for a sparsity over 99% when the data matrix is stored in COO mode and 95% when the data is stored in CSR mode.
* Restoring mode: use torch.sparse when necessary in forward and templately restore the matrix into dense mode in backward. This mode is more flexible and can be used in most cases.

So far, the tensors are all stored in COO format, since COO format support the majority of the operations. However, the CSR format is more efficient in matrix-vector multiplication, so we plan to add the CSR format in the future. This asks for a customized CUDA-kernel support for Hadamard product for CSR matrix and dense matrix.

## Future Plan

* [ ]  optimize the code to support CSR representation
