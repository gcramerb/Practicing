#include <iostream>
#include <cstdint> // for uint64_t
#include <map>
#include <string>   
void a(){
        uint64_t x = 10000000000ULL;
    uint64_t y = 20000000000ULL;
    uint64_t result = x+ y;
}

int main() {

for(int i =0; i< 2; i++){
        std::cout << i++;
        std::cout << i;
    }

}