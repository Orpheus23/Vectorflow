#include <algorithm>
#include <cuda.h>
#include <curand.h>
#include <stdio.h>

template <typename T>
struct tensor
    {
        T* flattened;
        long long int *shape;
        long long int *stride;

        
        __device__ __host__ T operator() (const int j , const int dims, const long long int* ref_stride)
        {
            int result = 0;
            for (int i = 0;i<dims;i++)
            {
                long long int curr_index = ref_stride[i];
                int idxs = (j/ (1*(curr_index==1)+(curr_index)*(curr_index!=1)));
                long long int stride_i = *(stride+i),shape_i=*(shape+i);
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
        
        __host__ __device__ void print_elems(const int total_length)
        {
            for (int i = 0;i<total_length;i++)
            {
                cout<< *(flattened+i) << " ";
            }
            cout << "\n";    
        }

    };