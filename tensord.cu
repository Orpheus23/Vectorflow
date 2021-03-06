#include "Core/Tensor.hpp"
using namespace std;


void verify_result(std::vector<int> &a, std::vector<int> &b,
                   std::vector<int> &c) {
  for (int i = 0; i < a.size(); i++) {
    assert(c[i] == a[i] + b[i]);
  }
}


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
    a0.print_dim();
    a0.print_stride();
    int value = a0(2,0);
    //vector <int> idxs {1,1};
    cout<<value<<endl;
    Tensor<int> a2(v);
    
    a2.print_elems();
    a2.print_dim();
    a2.print_stride();

    Tensor<int> a3; 
    //a3.print_elems();
    a3 = a2+a0;
    a3.print_elems();
    
    
    Tensor <int> a4(v);
    a4.concat(a3, 1);
    a4.print_elems();
    vector <int> shape {4,4};




    Tensor <float> a5;
    a5.random_initialize(shape);
    std::cout << "Printing elements of a5" << std::endl;
    a5.print_stride();
    a5.print_dim();
    a5.dim_space();
    a5.print_elems();
    


    Tensor <float,4,4> b1;
    Tensor <float> b2;
    
    b1.random_initialize();
    
    vector< vector<int> > slicey{{1,3},{},{1,3}};
    b2 = b1.slice(slicey);
    b1.print_elems();
    b2.print_dim();
    b2.print_elems();
    
    
    
    

}

