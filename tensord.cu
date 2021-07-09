#include <vector>
#include <iostream>
#include <numeric>
#include <initializer_list>
#include <algorithm>
#include <cuda.h>
#include <curand.h>
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


 

template <typename T>
vector<T> return_row (vector<T> v,vector<T> empty,vector<long long int> &dims)
{
    empty.insert(empty.end(),v.begin(),v.end());
    dims.push_back(v.size());
    return empty;
}


template<class T,typename T_>
vector<T_> return_row(vector<T> &rows,vector<T_> empty,vector<long long int> &dims)
{
    dims.push_back(rows.size());
    for (T element:rows)
    {   
        vector <T_> empty_temp = return_row(element,empty,dims);
        empty.insert( empty.end(), empty_temp.begin(),empty_temp.end());
    }
    
    return empty;
}




template<typename T>
class Tensor
{   
    
    int is_gpu = 0;
    private:       
        
        
        vector<long long int> stride_convert(vector<long long int> stride_vector_o)
        {
            stride_vector_o[stride_vector_o.size()-1]=1;
            for (int i = stride_vector_o.size()-1;i>0;i--)
                {
                    stride_vector_o[i-1]*=stride_vector_o[i];
                }
            
            return stride_vector_o;
        }
        
        vector<long long int> dimension_list;
        vector<long long int> stride_vector;
        const long long int shape_total;
        const int N;
        //thrust::host_vector <T> tensor;
        vector <T> tensor_cpu;
        tensor <T> mat;

    
    public:
        //Initialize the Constructer for a vector without shape given in the vector and explicitely giving shape
        template<class Y,typename ... Args>
        Tensor(Y a,Args... Axii)
        {
            dimension_list = initializer_list<long long int>{Axii...};
            stride_vector = stride_convert(dimension_list);
            //stride_vector = dimension_list;
            shape_total = accumulate(dimension_list.begin(),dimension_list.end(),1,multiplies<long long int>());
            N = dimension_list.size();
            vector <long long int> temp;
            tensor_cpu = return_row(a,tensor_cpu,temp);
            del temp;
        }

        //Initialize the Constructor for multidimensional array *infers shape from that*
        template<class Y>
        Tensor(Y a)
        {
            tensor_cpu = return_row(a,tensor_cpu,dimension_list);
            stride_vector = stride_convert(dimension_list);
            shape_total = accumulate(dimension_list.begin(),dimension_list.end(),1,multiplies<long long int>());
            N = dimension_list.size();
        }

        //Initialize the Constructor for only shape *defaults to zero vector*
        template<typename ... Args>
        Tensor(Args... Axii)
        {
            dimension_list = initializer_list<long long int>{Axii...};
            stride_vector = stride_convert(dimension_list);
            //stride_vector = dimension_list;
            shape_total = accumulate(dimension_list.begin(),dimension_list.end(),1,multiplies<long long int>());
            N = dimension_list.size();
            vector<T> tens(shape_total,(T)0);
            tensor_cpu = tens;
        }

        //Initialize the Constructor for no inputs *defaults to zero vector*
        Tensor()
        {
            vector<T> tens(1,(T)0);
            tensor_cpu = tens;
            
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
        
        //Reduces it to a base tensor class for gpu
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
                result+= (index[i]%dimension_list[i])*stride_vector[i];
            }
            return tensor_cpu[result]; 
        }

        Tensor<T> operator()(vector< vector<int> > index)
        { 
            int result;
            vector <T> vals;
            for (int i = 0;i<index[0].size();i++)
            {
                result = 0
                for (int j = 0; j<index.size();j++)
                {
                    result+= (index[j][i]%dimension_list[j])*stride_vector[j];
                }
                vals.push_back(tensor_cpu[result])
            }
            Tensor <T> rets(vals);
            return rets; 
        }

        Tensor<T> slice(vector< vector<int> > Axii)
        {
            
            
            vector <long long int> idxs(1,0);
            vector <T> vals;
            vector <int> sh;
            for (int i = Axii.size()-1;i>=0;i--)
            {
                auto old_size = idxs.size();
                int current = Axii[i][0];
                int end = Axii[i][Axii[i].size()-1];
                int reverse = (end-current)<0;
                reverse = (1-2*reverse)-(end==current);
                if (reverse!=0) sh.push_back(reverse *(end-current));
                reverse += (end==(current-1))-((end-1)==current);
                
                idxs.reserve(((reverse *(end-current)))* (old_size));
                std::copy_n(idxs.begin(), old_size, std::back_inserter(idxs));
                
                for (int j = 0;j<idxs.size();j++)
                {
                    idxs[j]+=(current%dimension_list[i])*stride_vector[i]);
                    current += reverse*((j%old_size)==0);
                    if (i == 0) vals.push_back(tensor_cpu[idxs[j]]);
                    
                }
                
            }
            Tensor <T> result(vals);
            result.reshape(sh);
            return result;


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


        

        Tensor operator + (Tensor &b)
        {
            b.to_gpu();
            to_gpu();
            Tensor<T> output_tensor();
            output_tensor.to_gpu();
            
            tensor<T> mat1;
            tensor<T> mat2;
            tensor<T> mat3;
            mat1 = b.basic();
            mat2 = basic();
            mat3 = output_tensor.basic();

            add <T> <<<(shape_total + 4  - 1) / 4 , 4>>> (mat1,mat2,mat3,shape_total,dimension_list.size());
            //cudaDeviceReset(); conflicts
            b.to_cpu();
            to_cpu();
            output_tensor.to_cpu();

            cout << "CUDA Add successful";
            cout << endl;
            
            return output_tensor;
        }

         Tensor operator * (Tensor &b)
        {
            b.to_gpu();
            to_gpu();
            Tensor<T> output_tensor();
            output_tensor.to_gpu();
            
            tensor<T> mat1;
            tensor<T> mat2;
            tensor<T> mat3;
            mat1 = b.basic();
            mat2 = basic();
            mat3 = output_tensor.basic();

            dot <T> <<<(shape_total + 4  - 1) / 4 , 4>>> (mat1,mat2,mat3,shape_total,dimension_list.size());
            //cudaDeviceReset();
            b.to_cpu();
            to_cpu();
            output_tensor.to_cpu();

            cout << "CUDA Add successful";
            cout << endl;
            
            return output_tensor;
        }

        //Transpose (Basically affects the strides)
        void Transpose(vector <int> Axis = {1})
        {
            long long int temp,temp_d;
            vector <long long int> stride2 = stride_vector;
            for (int i = 0;i<Axis.size();i++)
                {
                    temp = stride_vector[i];
                    stride_vector[i] = stride_vector[Axis[i]];
                    stride_vector[Axis[i]] = temp;
                    temp_d = dimension_list[i];
                    dimension_list[i] = dimension_list[Axis[i]];
                    dimension_list[Axis[i]] = temp_d;
                }
        }
        
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

        void zeros(vector <int> Shape)
        {
            vector<long long int> dimension_list (Shape.begin(), Shape.end());
            vector<long long int> stride_vector = stride_convert(dimension_list);
            //stride_vector = dimension_list;
            const long long int shape_total = accumulate(dimension_list.begin(),dimension_list.end(),1,multiplies<long long int>());
            const int N = dimension_list.size();
            vector<T> tens(shape_total,(T)0);
            tensor_cpu = tens;

        }
        
        

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
            cudaMalloc(&mat.stride, sizeof(long long int)*dimension_list.size());
            cudaMalloc(&mat.shape, sizeof(long long int)*dimension_list.size());
            cudaMemcpy(mat.flattened, tensor_cpu.data(), sizeof(T)*shape_total, cudaMemcpyHostToDevice);
            cudaMemcpy(mat.shape, dimension_list.data(), sizeof(long long int)*dimension_list.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(mat.stride, stride_vector.data(), sizeof(long long int)*stride_vector.size(), cudaMemcpyHostToDevice);
            is_gpu = 1;
            
        }
        
        __host__ void to_cpu()
        {
            cudaMemcpy(tensor_cpu.data(), mat.flattened, sizeof(T)*shape_total, cudaMemcpyDeviceToHost);
            
            cudaFree(mat.flattened);
            cudaFree(mat.shape);
            //mat = {};
            is_gpu = 0;
        }
        
        void expand_dims(int Axis = 0)
        {
            auto itPos = stride_vector.begin() + Axis;
            // Insert element with value 9 at 4th Position in vector
            stride_vector.insert(itPos, 9);
        }

        //Reshape (Again just affects the strides)
        void Reshape(vector <int> Shape)
        {
            stride_vector = stride_convert(Shape);
        }


        
        vector<long long int> shape()
        {
            return dimension_list;
        }

        //concats two vectors along a given axis
        void concat (Tensor &b, int axis)
        {   
            long long int new_shape = (shape_total/dimension_list[axis])*(dimension_list[axis]+b.shape()[axis]);
            long long int orig_dim = dimension_list[axis];
            long long int sub_index = 0;
            dimension_list[axis] += b.shape()[axis];
            
            long long int opp_idx = 0;
            vector<T> output_tensor(new_shape,0);
            for (long long i = 0; i< new_shape;i++)
            {
                if (((i/stride_vector[axis])%dimension_list[axis] ) < orig_dim )
                {
                    output_tensor[i] = tensor_cpu[i-opp_idx];
                    sub_index+=1;
                }
                else
                {
                    output_tensor[i] = b[i-sub_index];
                    opp_idx +=1;
                    
                }
               
            }
            cout <<endl;
            stride_vector = stride_convert(dimension_list);
            tensor_cpu = output_tensor;


        }
};


int main()
{
    vector<vector<int> >
        v{  { 1, 2, 3, 4},
            { 3, 4, 5, 6},
            { 4, 5, 6, 7},
            { 7, 8, 9, 10}  };

    
    //2 5 7 11 5 8 10 14 7 10 6 16 11 14 16 10
    Tensor<int> a0(v); 
    a0.print_elems();
    int value = a0(2,0);
    //vector <int> idxs {1,1};
    value = a0(2,0);
    cout<<value<<endl;
    Tensor<int> a2(v);
    
    a2.print_elems();
    Tensor<int> a3(4,4); 
    a3 = a2+a0;
    Tensor <int> a4(v);
    a4.concat(a3, 1);
    a4.print_elems();
    a3.print_elems();
    Tensor <float> a5;
    a5.random_initialize();
    a5.print_elems();
    Tensor <float> b1,b2;
    vector <int> shape {8,8,8};
    b1.zeros(shape);
    vector< vector<int> > slicey{{2,4},{3,5},{4,7}};
    b2 = b1.slice(slicey);
    b1.print_elems();
    b2.print_elems();
    
    

}

