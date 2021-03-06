
#include <iostream>
#include <cuda.h>
#include <curand.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

template <typename T>
struct tensor
    {
        T* flattened;
        int *shape;
        int *stride;

        
        __device__ __host__ T operator() (const int j , const int dims, const int* ref_stride)
        {
            int result = 0;
            for (int i = 0;i<dims;i++)
            {
                int curr_index = ref_stride[i];
                int idxs = (j/ (1*(curr_index==1)+(curr_index)*(curr_index!=1)));
                int stride_i = *(stride+i),shape_i=*(shape+i);
                result += (stride_i) * (idxs%(shape_i));
            }
            
            return *(flattened+result); 
        }
        
        __device__ __host__ T operator[](const int i)
        {
            return *(flattened+i);
        }
        
        __device__ __host__ void set(const int i, T j)
        {
            *(flattened+i) = j;
        }
        
        __device__ __host__ void set(const int *index,int i ,int* basic_shape, const int dims, T j)
        {
            int result = 0;
            for (int i = 0;i<dims;i++)
            {
                result+= index[i]*(*(stride+i));
            }
            *(flattened+result) = j;
        }
        
        __host__ __device__ void print_elems(int total_length)
        {
            
            printf("%d \n",flattened[2]);
            
        }

    };