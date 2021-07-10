#include <vector>
#include <iostream>

template <typename T>
std::vector<T> Flatten_with_dims(vector<T> v,vector<T> empty,vector<long long int> &dims,int level)
{
    vector <T> temp;
    temp.insert(temp.end(),v.begin(),v.end());
    if (dims.size()==level) 
    {
        dims.push_back(rows.size());
    }
    return temp;
}


template<class T,typename T_>
std::vector<T_> Flatten_with_dims(vector<T> &rows,vector<T_> empty,vector<long long int> &dims,int &level)
{
    
    if (dims.size()==level) 
    {
        dims.push_back(rows.size());
    }
    vector <T_> empty_temp ;
    for (T element:rows)
    {   
        empty_temp = Flatten_with_dims(element,empty,dims,level+1);
        
        empty.insert( empty.end(), empty_temp.begin(),empty_temp.end());
    }
    
    return empty;
}