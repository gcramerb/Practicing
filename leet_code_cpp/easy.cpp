#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>

class Solution121 {
public:
    int maxProfit(std::vector<int>& prices) {
        if(prices.size() < 2){return 0;}
        size_t buyDay = 0, sellDay = 1;
        int profit = 0;
        while(buyDay <= prices.size() -1 && sellDay <= prices.size() -1){
            if(prices[sellDay] > prices[buyDay]){
                int current_profit = prices[sellDay] - prices[buyDay];
                if(current_profit > profit){
                    profit = current_profit;
                }
                sellDay++;
            }else{
                buyDay++;
            }
            if(sellDay <= buyDay){
                sellDay = buyDay + 1;
            }
            
        }
        return profit;
    }
};
void runTest121(){
    Solution121 sol;
    std::vector<int> prices = {7,1,5,3,6,4};
    int result = sol.maxProfit(prices);
    std::cout << "Max Profit: " << result << std::endl;
}
class Solution1 {
public:
    int binary_search_index(const std::vector<std::pair<int, int>>& v, int target,int originalIndex) {
    std::pair<int,int> targetPair{target,originalIndex};
    auto it = std::lower_bound(v.begin(), v.end(), targetPair,[](const std::pair<int, int>& p, std::pair<int,int> val) {
            return p.first == val.first && p.second != val.second;
        });
    if (it != v.end()) {
        return it - v.begin();  // Retorna o índice
    }
    return -1;  // Não encontrado
}
    std::vector<int> twoSum(std::vector<int>& nums, int target) {
        size_t N = nums.size();
        if(N == 2){
            return {0,1};
        }
        std::vector<std::pair<int, int>> indexed;
        for (int i = 0; i < N; i++) {
            indexed.push_back({nums[i], i});
        }
        std::sort(indexed.begin(), indexed.end());
        
        for(int i = 0; i < N; i++){
            int scnd_idx = binary_search_index(indexed,target - indexed[i].first, indexed[i].second); 
            if(scnd_idx  > 0 && scnd_idx != i){
                return {indexed[i].second, indexed[scnd_idx].second};
            }
        }
        return {-1,-1};

    }
};
void runTest1(){
    Solution1 sol;
    std::vector<int> nums = {0,4,3,0};
    int target = 6;
    std::vector<int> result = sol.twoSum(nums,target);
    std::cout << "Indices: [" << result[0] << ", " << result[1] << "]" << std::endl;
}
int main() {

    runTest1();
    return 0;
}