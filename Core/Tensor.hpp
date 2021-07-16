
#include "../cudaflow/kernels.cuh"
#include "Flatten.hpp"
#include <vector>
#include <iostream>
#include <initializer_list> 
#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <numeric>

template<typename T, std::size_t... Types>
class Tensor
{   
    
    int is_gpu = 0;
    private:       
        
       
        std::vector<int> stride_convert(std::vector<int> stride_vector_o)
        {
            stride_vector_o[stride_vector_o.size()-1]=1;
            for (int i = stride_vector_o.size()-1;i>0;i--)
                {
                    stride_vector_o[i-1]*=stride_vector_o[i];
                }
            
            return stride_vector_o;
        }
        
        
        //thrust::host_vector <T> tensor;
        std::vector <T> tensor_cpu;
        tensor <T> mat;

        std::vector<int> dimension_list = std::initializer_list<int>{Types...};
        //std::vector<int> stride_vector = stride_convert(dimension_list);
        //stride_vector = dimension_list;
        //int shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());
        //int N = dimension_list.size();

        //std::vector<int> dimension_list=std::initializer_list<int>{Types...} ;
        std::vector<int> stride_vector;
        int shape_total;
        int N;

    
    public:

        //Initialize the Constructor for multidimensional array *infers shape from that*
        template<class Y>
        Tensor(Y a)
        {
            int idx = 0;
            std::vector <int> dim2 = {};            
            tensor_cpu = Flatten_with_dims(a,tensor_cpu,dim2,idx);
            
            if (dimension_list.size()==0) dimension_list = dim2;
            stride_vector = stride_convert(dimension_list);
            shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());
            N = dimension_list.size();
            
            
        }
        template <typename T_b,std::size_t...Types_b>
        Tensor(Tensor<T_b,Types_b...> &b)
        {
            dimension_list = b.shape();
            is_gpu = b.is_gpu;
            tensor_cpu = b.flatten();
            stride_vector = stride_convert(dimension_list);
            shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());
            N = dimension_list.size();

        }

        //Initialize the Constructor for no inputs *defaults to zero std::vector*
        Tensor()
        {
            dimension_list.clear();
            dimension_list = std::initializer_list<int>{Types...};
            if (dimension_list.size()==0) dimension_list.push_back(1);
            stride_vector = stride_convert(dimension_list);
            shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());
            N = dimension_list.size();

            std::vector<T> tens(shape_total,(T)0);
            tensor_cpu = tens;
            
        }

        ~Tensor()
        {
            dimension_list.clear();
            std::vector<int>().swap(dimension_list);
            stride_vector.clear();
            std::vector<int>().swap(stride_vector);
            tensor_cpu.clear();
            std::vector<T>().swap(tensor_cpu);
            
        }

        //Prints the dimensions of the std::vector as created during declaration
        void print_dim()
        {
            std::cout<<"[ ";
            for (int element:dimension_list)
                std::cout<< element <<" ";
            std::cout<<"]"<<std::endl;
        }

        //Prints the product of all dims, remains constant while reshaping and transpose
        void dim_space()
        {
            std::cout<< "Shape space: " <<shape_total << std::endl;
        }

        //Prints the elements as they are stored during computation ie. a 1D std::vector
        void print_elems()
        {
            std::vector<int> open (dimension_list.size(),0);
            int brack = dimension_list[dimension_list.size()-1];
            int val = 0;
            std::cout<<"Printing "<<shape_total<<" elements:- ";
            for (int i = 0; i<shape_total;i++)
            {
                for(int j = 0; j<dimension_list.size();j++)
                {
                    val = (stride_vector[j]*brack);
                    if ((i%val)==0)
                    {
                        if(open[j]==0)
                        {
                            std::cout<<"[";
                            open[j] = 1;
                        }
                        else
                        {
                            std::cout<<"],[";
                        }
                        
                    }
                    
                }
                std::cout<< tensor_cpu[i] <<","+(8)*((i+1)%(val)==0);
            }
            for (int i = 0; i <N;i++)
            {
                if (open[i] !=0)
                {
                    std::cout <<"]";
                }
            }
            
            std::cout<<std::endl;
        }
        
        //Reduces it to a base tensor class for gpu
        auto cuda()
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
            std::vector<int> index {Axii...};
            int result = 0;
            for (int i = 0;i<index.size();i++)
            {
                result+= (index[i]%dimension_list[i])*stride_vector[i];
            }
            return tensor_cpu[result]; 
        }

        Tensor<T> operator()(std::vector< std::vector<int> > index)
        { 
            int result;
            std::vector <T> vals;
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

        Tensor<T> slice(std::vector< std::vector<int> > Axii)
        {
            
            
            std::vector <T> idxs(1,(T)0);
            std::vector <int> sh;
            auto old_size = 0;
            int current = 0;
            int end  = 0;
            int reverse = 0;
            int S = !(Axii.size()==N);
            for (int i = Axii.size()-1;i>=0;i--)
            {
                
                old_size = idxs.size();
                if (Axii[i].empty())
                {
                    current = 0;
                    end = 0;
                }
                else
                {
                    current = (dimension_list[i-S]+Axii[i][0])%dimension_list[i-S];
                    end = (dimension_list[i-S]+Axii[i][Axii[i].size()-1])%dimension_list[i-S];
                    
                }
                
                reverse = 1-(2*((end-current)<0))-(end==current);
                sh.push_back(reverse *(end-current) + Axii[i].empty());
                //reverse += (end==(current-1))-((end-1)==current);
                
                //idxs.reserve(((reverse *end)-(reverse*current)-1)* (old_size));
                idxs.resize(((reverse *end)-(reverse*current))* (old_size) + old_size*(Axii[i].empty()));
                
                end-=(reverse>0);
                
                for (int j = idxs.size()-1;j>=0;j--)
                {
                    idxs[j] = idxs[j%old_size] + (end%dimension_list[i-S])*stride_vector[i-S];
                    end -= reverse*(((j+1)%(old_size+1))==0);
                    idxs[j] = tensor_cpu[(int)idxs[j]]*(i==0) + (i!=0)*idxs[j];
                }
                S += ((S<=0)-(S>0))*Axii[i].empty();
                
            }
            Tensor <T> result(idxs);
            result.Reshape(sh);

            return result;


        }

        T operator[](const int i)
        {
            return tensor_cpu[i];
        }

        template <std::size_t...Types_b>
        void operator = (Tensor<T,Types_b...> b)
        {
            dimension_list = b.shape();
            is_gpu = b.is_gpu;
            tensor_cpu = b.flatten();

        }

        template <class B>
        void operator = (std::vector<B> b)
        {
            tensor_cpu = Flatten_with_dims(b,tensor_cpu,dimension_list,0);
            stride_vector = stride_convert(dimension_list);
            shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());
        }


        
        template <std::size_t ... Types_b>
        Tensor operator + (Tensor<T,Types_b...> &b)
        {
            b.to_gpu();
            to_gpu();
            Tensor<T,Types...> output_tensor;
            output_tensor.zeros(dimension_list);
            output_tensor.to_gpu();
            
            tensor<T> mat1;
            tensor<T> mat2;
            tensor<T> mat3;
            mat1 = b.cuda();
            mat2 = cuda();
            mat3 = output_tensor.cuda();
            //mat1.print_elems(shape_total);
            add <T> <<<(shape_total + 4  - 1) / 4 , 4>>> (mat1,mat2,mat3,shape_total,dimension_list.size());
            //cudaDeviceReset(); conflicts
            b.to_cpu();
            to_cpu();
            output_tensor.to_cpu();

            std::cout << "CUDA Add successful";
            std::cout << std::endl;
            
            return output_tensor;
        }

        template <std::size_t ... Types_b>
        Tensor operator * (Tensor<T,Types_b...> &b)
        {
            b.to_gpu();
            to_gpu();
            Tensor<T,Types...> output_tensor;
            output_tensor.zeros(dimension_list);
            output_tensor.to_gpu();
            
            tensor<T> mat1;
            tensor<T> mat2;
            tensor<T> mat3;
            mat1 = b.cuda();
            mat2 = cuda();
            mat3 = output_tensor.cuda();

            dot <T> <<<(shape_total + 4  - 1) / 4 , 4>>> (mat1,mat2,mat3,shape_total,dimension_list.size());
            //cudaDeviceReset();
            b.to_cpu();
            to_cpu();
            output_tensor.to_cpu();

            std::cout << "CUDA Add successful";
            std::cout << std::endl;
            
            return output_tensor;
        }

        //Transpose (cudaally affects the strides)
        void Transpose(std::vector <int> Axis = {1})
        {
            int temp,temp_d;
            std::vector <int> stridstride_vector;
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
        
        template <typename V>
        void random_initialize(std::vector <V> Shape)
        {
            dimension_list = Shape;
            stride_vector = stride_convert(dimension_list);
            shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());
            N = dimension_list.size();
            random_initialize();
            dimension_list = Shape;
        }

        void random_initialize()
        {
            T* return_data;
            curandGenerator_t gen;
            
            cudaMalloc((void **)&return_data, shape_total*sizeof(T));

            curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
            
            curandSetPseudoRandomGeneratorSeed(gen,1234ULL);

            curandGenerateUniform(gen, return_data, shape_total);

            cudaMemcpy(tensor_cpu.data(), return_data, shape_total*sizeof(T),cudaMemcpyDeviceToHost);
            curandDestroyGenerator(gen);
            cudaFree(return_data);
        }

        template <typename V>
        void zeros(std::vector <V> Shape)
        {
            dimension_list = Shape;
            stride_vector = stride_convert(dimension_list);
            shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());
            N = dimension_list.size();
            std::vector<T> tens(shape_total,(T)0);
            tensor_cpu = tens;

        }
        
        

        void print_stride()
        {
            std::cout<<"[ ";
            for (auto element:stride_vector)
                std::cout<< element <<" ";
            std::cout<<"]"<<std::endl;
        }


        __host__ void to_gpu()
        {
            
            cudaMalloc(&mat.flattened, sizeof(T)*shape_total);
            cudaMalloc(&mat.stride, sizeof(int)*dimension_list.size());
            cudaMalloc(&mat.shape, sizeof(int)*dimension_list.size());
            cudaMemcpy(mat.flattened, tensor_cpu.data(), sizeof(T)*shape_total, cudaMemcpyHostToDevice);
            cudaMemcpy(mat.shape, dimension_list.data(), sizeof(int)*dimension_list.size(), cudaMemcpyHostToDevice);
            cudaMemcpy(mat.stride, stride_vector.data(), sizeof(int)*stride_vector.size(), cudaMemcpyHostToDevice);
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
            // Insert element with value 9 at 4th Position in std::vector
            stride_vector.insert(itPos, 9);
        }

        //Reshape (Again just affects the strides)
        void Reshape(std::vector <int> Shape)
        {
            stride_vector = stride_convert(Shape);
            dimension_list = Shape;
            N = dimension_list.size();
        }


        
        std::vector<int> shape()
        {
            return dimension_list;
        }

        //concats two std::vectors along a given axis
        template<std::size_t ... Types_b>
        void concat (Tensor<T,Types_b...> &b, int axis)
        {   
            int new_shape = (shape_total/dimension_list[axis])*(dimension_list[axis]+b.shape()[axis]);
            int orig_dim = dimension_list[axis];
            int sub_index = 0;
            dimension_list[axis] += b.shape()[axis];
            int opp_idx = 0;
            std::vector<T> output_tensor(new_shape,0);
            for (int i = 0; i< new_shape;i++)
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
            stride_vector = stride_convert(dimension_list);
            tensor_cpu = output_tensor;
            shape_total = std::accumulate(dimension_list.begin(),dimension_list.end(),1,std::multiplies<int>());


        }
};