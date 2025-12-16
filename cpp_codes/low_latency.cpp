#include <iostream>
#include <cstdint> // for uint64_t
#include <map>
#include <string>   

// Example template function that works with uint64_t
template<typename T>
T add(T a, T b) {
    return a + b;
}
enum class Side {Bid, Ask};
using OrderId = uint64_t;
using Price = uint64_t;
using Quantity = int64_t;
void addOrder(Side side, OrderId orderId, Price price, Quantity quantity) {
    // Function to add an order
    std::cout << "Adding order: Side=" << (side == Side::Bid ? "Bid" : "Ask")
              << ", OrderId=" << orderId
              << ", Price=" << price
              << ", Quantity=" << quantity << std::endl;
}
void ModifyOrder(OrderId orderId,  Quantity newQuantity) {
    // Function to modify an order
    std::cout << ", OrderId=" << orderId
              << ", Quantity=" << newQuantity << std::endl;
}
void DeleteOrder(OrderId orderId) {
    // Function to delete an order
    std::cout << "Deleting order with OrderId=" << orderId << std::endl;
}
int main() {
    uint64_t x = 10000000000ULL;
    uint64_t y = 20000000000ULL;
    uint64_t result = add<uint64_t>(x, y);

    std::map<Price,Quantity,std::greater<Price>> mBidLevels;
    std::map<Price,Quantity,std::less<Price>> mAskLevels;


    std::cout << "Result: " << result << std::endl;
    return 0;
}