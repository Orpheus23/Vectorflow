
#include "kernels.cuh"
#include "tensor.cuh"
#include "Flatten.hpp"
#include <vector>
#include <iostream>
#include <initializer_list> 
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>

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
        
        vector<long long int> dimension_list = {};
        vector<long long int> stride_vector;
        long long int shape_total=1;
        int N=1;
        //thrust::host_vector <T> tensor;
        vector <T> tensor_cpu;
        tensor <T> mat;

    
    public:

        //Initialize the Constructor for multidimensional array *infers shape from that*
        template<class Y>
        Tensor(Y a)
        {
            N = dimension_list.size();
            tensor_cpu = Flatten_with_dims(a,tensor_cpu,dimension_list,0);
            stride_vector = stride_convert(dimension_list);
            shape_total = accumulate(dimension_list.begin(),dimension_list.end(),1,multiplies<long long int>());
            
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
                result = 0;
                for (int j = 0; j<index.size();j++)
                {
                    result+= (index[j][i]%dimension_list[j])*stride_vector[j];
                }
                vals.push_back(tensor_cpu[result]);
            }
            Tensor <T> rets(vals);
            return rets; 
        }

        Tensor<T> slice(vector< vector<int> > Axii)
        {
            
            
            vector <long long int> idxs(1,0);
            vector <T> vals;
            vector <long long int> sh;
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
                    idxs[j] += (current%dimension_list[i])*stride_vector[i];
                    current += reverse*((j%old_size)==0);
                    if (i == 0) vals.push_back(tensor_cpu[idxs[j]]);
                    
                }
                
            }
            Tensor <T> result(vals);
            result.Reshape(sh);
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
            Tensor<T> output_tensor;
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
            Tensor<T> output_tensor;
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
        void Reshape(vector <long long int> Shape)
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