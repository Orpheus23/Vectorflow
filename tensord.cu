#include <vector>
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <queue>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
using namespace std;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
    #define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
  for (int i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}

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
                
                int idxs = (j/ (1*(ref_stride[i]==1)+(ref_stride[i])*(ref_stride[i]!=1)));
                result += (*(stride+i))*(idxs%(*(shape+i)));
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




template <typename T>
__global__ 
void add(tensor<T> a, tensor<T> b,tensor<T> c ,int N_,int dims) 
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


 

template <typename T,typename T_>
vector<T_> return_row (const T &v,vector<T_> empty,const vector<int> &dims)
{
    return vector<T_>(1, v);
}


template<class T,typename T_>
vector<T_> return_row(vector<T> &rows,vector<T_> empty,const vector<int> &dims)
{
    if (dims.size()>1)
    {
        for (T element:rows)
        {   
            vector <int> new_dims(dims.begin()+1,dims.end());
            empty = return_row(element,empty,new_dims);

            //empty.insert( empty.end(), empty_temp.begin(),empty_temp.end());
        }
    }
    else
    {
        for (T element:rows)
        {
            vector <int> new_dims(dims.begin()+1,dims.end());
            vector<T_> empty_temp = return_row(element,empty,new_dims);

            empty.insert( empty.end(), empty_temp.begin(),empty_temp.end());
        }
    }
    return empty;
}


template<typename T, size_t ... Types>
class Tensor
{   
    
    int is_gpu = 0;
    private:       
        
        vector<int> dimension_list = initializer_list<int>{Types...};
        vector<int> stride_vector = stride_convert(dimension_list);
        //stride_vector = dimension_list;
        const int shape_total = accumulate(dimension_list.begin(),dimension_list.end(),1,multiplies<int>());
        const int N = dimension_list.size();
        vector<T> stride_convert(vector<int> stride_vector_o)
        {
            stride_vector_o[stride_vector_o.size()-1]=1;
            for (int i = stride_vector_o.size()-1;i>0;i--)
                {
                    stride_vector_o[i-1]*=stride_vector_o[i];
                }
            
            return stride_vector_o;
        }
        
        
        //thrust::host_vector <T> tensor;
        vector <T> tensor_cpu;
        tensor <T> mat;

    
    public:
        //Initialize the Constructer for nested vectors
        template<class Y>
        Tensor(Y a)
        {
            
            tensor_cpu = return_row(a,tensor_cpu,dimension_list);
        }

        //Initialize the Constructor for no inputs *defaults to zero vector*
        Tensor()
        {
            tensor_cpu(shape_total,0);
            
        }

        //Prints the dimensions of the vector as created during declaration
        void print_dim()
        {
            cout<<"[ ";
            for (int element:dimension_list)
                cout<< element <<" ";
            cout<<"]"<<endl;
        }

        //Prints the product of all dims, remains constant while reshaping and transpose
        void dim_space()
        {
            cout<< "Shape space: " <<shape_total << endl;
        }

        //Prints the elements as they are stored during computation ie. a 1D vector
        void print_elems()
        {
            cout<<"Printing elements:- ";
            cout<<"[ ";
            for (auto elem:tensor_cpu)
                cout<< elem<<" ";
            cout<<"]"<<endl;
        }
        
        //template <class Output>
        auto basic()
        {
            assert(is_gpu == 1);

            if (is_gpu ==1)
            {
                return mat;
            }
        }

        auto flatten()
        {
            return tensor_cpu;
        } 


        template <typename ... Args>
        T operator()(const Args... Axii)
        { 
            vector<int> index {Axii...};
            int result = 0;
            for (int i = 0;i<index.size();i++)
            {
                result+= index[i]*stride_vector[i];
            }
            return tensor_cpu[result]; 
        }

        T operator[](const int i)
        {
            return tensor_cpu[i];
        }


        void operator = (Tensor b)
        {
            dimension_list = b.shape();
            is_gpu = b.is_gpu;
            tensor_cpu = b.flatten();

        }


        

        Tensor operator + (Tensor b)
        {
            b.to_gpu();
            to_gpu();
            Tensor<T,Types ...> output_tensor();
            output_tensor.to_gpu();
            
            tensor<T> mat1;
            tensor<T> mat2;
            tensor<T> mat3;
            mat1 = b.basic();
            mat2 = basic();
            mat3 = output_tensor.basic();
            //cudaMalloc(&mat3.flattened, sizeof(T)*shape_total);
            //cudaMalloc(&mat3.shape, sizeof(int)*dimension_list.size());
            add <T> <<<(shape_total + 4  - 1) / 4 , 4>>> (mat1,mat2,mat3,shape_total,dimension_list.size());
            //cudaDeviceReset();
            b.to_cpu();
            to_cpu();
            output_tensor.to_cpu();
            //cudaMemcpy(out, mat3.flattened, sizeof(T)*shape_total, cudaMemcpyDeviceToHost);
            //cudaFree(mat3.flattened);
            //cudaFree(mat3.shape);
            cout << "CUDA Add successful";
            cout << endl;
            
            //output_tensor.print_elems();
            //to_cpu();am seder
            //b.to_cpu();
            return output_tensor;
        }

        //Transpose (Basically affects the strides)
        void Transpose(vector <int> Axis = {1})
        {
            int temp;
            vector <int> stride2 = stride_vector;
            for (int i = 0;i<Axis.size();i++)
                {
                    temp = stride_vector[i];
                    stride_vector[i] = stride_vector[Axis[i]];
                    stride_vector[Axis[i]] = temp;
                    temp = dimension_list[i];
                    dimension_list[i] = dimension_list[Axis[i]];
                    dimension_list[Axis[i]] = temp;
                }
        }
        /*
        void random_initialize()
        {
            T* return_data;
            curandGenerator_t gen;
            cudaMalloc((void **)&return_data, shape_total*sizeof(T));

            curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
            
            curandSetPseudoRandomGeneratorSeed(gen,1234ULL);

            curandGenerateUniform(gen, return_data, shape_total);

            cudaMemcpy(&tensor_cpu[0], return_data, shape_total*sizeof(T),cudaMemcpyDeviceToHost);
            curandDestroyGenerator(gen);
            cudaFree(return_data);
        }
        */
        

        void print_stride()
        {
            cout<<"[ ";
            for (int element:stride_vector)
                cout<< element <<" ";
            cout<<"]"<<endl;
        }


        __host__ void to_gpu()
        {
            
            cudaMalloc(&mat.flattened, sizeof(T)*shape_total);
            cudaMalloc(&mat.stride, sizeof(int)*dimension_list.size());
            cudaMalloc(&mat.shape, sizeof(T)*dimension_list.size());
            for (auto elem:tensor_cpu)
            {
                cout << elem<< " ";
            }
            cudaMemcpy(mat.flattened, tensor_cpu.data(), sizeof(T)*shape_total, cudaMemcpyHostToDevice);
            cudaMemcpy(mat.shape, dimension_list.data(), sizeof(T)*dimension_list.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(mat.stride, stride_vector.data(), sizeof(int)*stride_vector.size(), cudaMemcpyHostToDevice);
            is_gpu = 1;
		    //cudaMemcpy(tensor_gpu, return_arr, sizeof(T)*tensor.size(), cudaMemcpyDeviceToHost);
            
        }
        
        __host__ void to_cpu()
        {
            /*
            tensor = tensor_gpu;
            is_gpu = 0;
            tensor_gpu.clear();
            */
            
            //tensor_cpu.clear();
            cudaMemcpy(tensor_cpu.data(), mat.flattened, sizeof(T)*shape_total, cudaMemcpyDeviceToHost);
            
            cout << endl;
            cudaFree(mat.flattened);
            cudaFree(mat.shape);
            //mat = {};
            is_gpu = 0;
            //thrust::device_vector <T>().swap(tensor_gpu); 
        }
        
        /*
        auto begin()
        {
            if (is_gpu == 0)
            {
                return tensor.begin();
            }
            else
            {
                return tensor_gpu.begin();
            }
            
        
        }
        
        auto end()
        {
            if (is_gpu == 0)
            {
                return tensor.end();
            }
            else
            {
                return tensor_gpu.end();
            }
            
        
        }

        */
        void expand_dims(int Axis = 0)
        {
            auto itPos = stride_vector.begin() + Axis;
            // Insert element with value 9 at 4th Position in vector
            stride_vector.insert(itPos, 9);
        }

        //Reshape (Again just affects the strides)
        void Reshape(vector <int> &Shape)
        {
            stride_vector = stride_convert(Shape);
        }


        
        vector<int> shape()
        {
            return dimension_list;
        }

        //concats two vectors along a given axis
        void concat (Tensor b, int axis)
        {   int new_shape = (shape_total/dimension_list[axis])*(stride_vector[axis]+b.shape()[axis]);
            vector<T> output_tensor(new_shape,0);
            for (int i = 0; i< new_shape;i++)
            {
                if (((i/stride_vector[axis])%dimension_list[axis] ) < dimension_list[axis] )
                {
                    output_tensor[i] = tensor_cpu[i];
                }
                else
                {
                    output_tensor[i] = b[i];
                }
               
            }
            tensor_cpu = output_tensor;
            dimension_list[axis] += b.shape()[axis];
            stride_vector = stride_convert(dimension_list);


        }
};



//template<typename T, size_t ... Types,typename... Args>
//T &Tensor<T,Types ...>::operator()(const Args ... Axis)

/*
template<typename T, size_t ... Types, size_t a1 , size_t b1 >
void concat(int axis, Tensor<T, Types... , a1 >& a,Tensor<T, Types..., b1 >& b)
{
    shape_a = a.shape();
    shape_b = b.shape();


}
*/

int main()
{
    vector<vector<int> >
        v{  { 1, 2, 3, 4},
            { 3, 4, 5, 6},
            { 4, 5, 6, 7},
            { 7, 8, 9, 10}  };

    
    //2 5 7 11 5 8 10 14 7 10 6 16 11 14 16 10
    Tensor<int,4,4> a0(v); 
    a0.print_elems();
    int value = a0(2,0);
    cout<<value<<endl;
    a0.print_stride();
    a0.Transpose();
    a0.print_elems();
    a0.print_stride();
    //vector <int> idxs {1,1};
    value = a0(2,0);
    cout<<value<<endl;
    Tensor<int,4,4> a2(v);
    
    a2.print_elems();
    Tensor<int,4,4> a3(); 
    a3 = a2+a0;
    Tensor <int,4,4> a4(v);
    a4.concat(a3, 0);
    a4.print_elems();
    a3.print_elems();
    
    

}
