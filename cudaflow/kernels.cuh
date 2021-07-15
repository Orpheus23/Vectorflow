#include "tensor.cuh"


//Kernel for element wise addition of two tensors
template <typename T>
__global__ 
void add(tensor<T> a, tensor<T> b,tensor<T> c ,long long int N_,int dims) 
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //__shared__ int idxs[];
    //printf("a value: %d, b value: %d\n", a[i], b[i]);
    if (i < N_)
    {
        c.set(i,a(i,dims,a.stride) + b(i,dims,a.stride));
        //printf("c value: %d, sum value: %d\n", c[i],a[i] + b[i]);
    }

    
}


//Kernel for element wise product of two tensors
template <typename T>
__global__ 
void dot(tensor<T> a, tensor<T> b,tensor<T> c ,int N_,int dims) 
{
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //__shared__ int idxs[];
    //printf("a value: %d, b value: %d\n", a[i], b[i]);
    if (i < N_)
    {
        c.set(i,a(i,dims,a.stride) * b(i,dims,a.stride));
        //printf("c value: %d, sum value: %d\n", c[i],a[i] + b[i]);
    }

    
}
