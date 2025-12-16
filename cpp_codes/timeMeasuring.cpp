
#include <chrono>
#include <iostream>
#include <limits>
int main(){
   
    uint64_t sum{0};
    uint64_t end = std::numeric_limits<uint64_t>::max()/100000000;
    auto start = std::chrono::high_resolution_clock::now();
    
    for(uint64_t i = 0; i< end;++i){
        sum+=i;
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << duration.count() ;
    return 0;
}