# My note during using libtorch
This note records my experience on libtorch

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
The most common bug in using libtorch is probably backwarding an in place operation. The most classic ones are the member functions ending with '_', as documented officialy. Also in official documentation are `+=`, `-=`, `*=`, `/=`

Problem:
* Say I have a tensor `x`, `x[i] = at::erf(x[i])` is in place so `a[i] = at::erf(x[i])` is the correct way of coding. However, in module `SAS` the similar pieces exist but why gradient works fine?
