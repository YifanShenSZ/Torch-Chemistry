# My note during using libtorch
This note records my experience on libtorch

## Surprising but making sense
`at::Tensor.symeig` returns eigenvectors in each column of a matrix, as mathematically required. I thought that would be in rows since it might call LAPACK, whose memory layout is a transpose to c++ frontend, but pytorch strides it back

## Shocking
For matrix `A`, `A += A.transpose(0, 1)` is different from `A = A + A.transpose(0, 1)`. The latter gives correct `A + A^T`, but the former one messes up by in-place adding and transposing at a same time, e.g. for 2 x 2 case it will perform `A[0][1] += A[1][0]; A[1][0] += A[0][1]`, which is wrong since the resulting `A[1][0]` is `A[0][1] + 2.0 * A[1][0]` rather than the desirable `A[0][1] + A[1][0]`

For matrix `A`, `A -= 2.0` is different from `A -= 2.0 * at::eye(A.size(0), A.options())`. The latter gives correct `A - 2.0`, but the former subtracts `2.0` from every element. That's fine though, as FORTRAN and numpy behave the same (probably because of broadcast semantics)

## Memory
Memory management is always a pain in the ass in using c++, so does libtorch

`at::Tensor` is a smart pointer, who releases memory based on an internal counter

`=` copies address rather than memory
* Q: Why this works if the right hand side is an expression?
* A: the return value of the right hand side expression has its own memory, and this memory is catched by the left hand side smart pointer

`from_blob` shares memory with the input pointer, so releasing that memory crashes the returned tensor:
* I'm not careless enough to delete that pointer
* But if that pointer is `std::vector<>.data()`, then it will be released when the `std::vector<>` goes out of scope

## In place operation
The most common bug in using libtorch is probably backwarding through an in place operation. The most classic ones are the member functions ending with '_', as documented officialy. Also in official documentation are `+=`, `-=`, `*=`, `/=`

Problem:
* Say I have a tensor `x`, `x[i] = at::erf(x[i])` turns out to be in-place so `a[i] = at::erf(x[i])` is the correct way of coding. ~~However, in module `SAS` the similar pieces exist but why gradient works fine?~~ That no longer works, so it must be my compiler (ifort 2019.4) who used to break the problematic in-place pieces

## Numerical instability
Explicitly looping over matrix elements deteriorates backward propagation. An example is `tchem::LA::UT_sy_U`
