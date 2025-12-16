#include <iostream>
#include <vector>
#include <map>
#include <array>
#include <algorithm>
#include <functional>
#include <queue>
using namespace std;


class Solution {
    private:
        using tuple = std::pair<long long,int>;
        priority_queue<int, std::vector<int>, std::greater<int>> m_avRooms;
        priority_queue<tuple, std::vector<tuple>, std::greater<tuple>> m_usedRooms;
        std::vector<int> m_roomsCount;
    public:
    void initRooms(int n){
         m_roomsCount.resize(n, 0);
        for (int i = 0; i < n; i++){
            m_avRooms.push(i);
        }
    }    
    void finishMeetings(const int current_time){
        int room_id;
        while(!m_usedRooms.empty() && m_usedRooms.top().first <= current_time){
            room_id = m_usedRooms.top().second;
            m_usedRooms.pop();
            m_avRooms.push(room_id);

        }
    }
    int maxUsedRoom(){
        int max_count = 0;
        int room_id = -1;
        for(int i = 0;i< m_roomsCount.size(); i++){
            if(m_roomsCount[i] > max_count){
                max_count = m_roomsCount[i];
                room_id = i;
            }
        }
        return room_id;
    }
    int mostBooked(int n, vector<vector<int>>& meetings) {
        int room_id;
        long long end_time;
        std::sort(meetings.begin(), meetings.end());
        initRooms(n);
        for(const auto& meeting : meetings){
            int start = meeting[0];
            finishMeetings(start);
            if(m_avRooms.empty()){
                int duration = meeting[1] - start;
                auto next_available = m_usedRooms.top();
                m_usedRooms.pop();
                room_id = next_available.second;
                end_time = next_available.first + duration;
            } else{
                room_id = m_avRooms.top();
                m_avRooms.pop();
                end_time = meeting[1];
            }
            m_usedRooms.push(tuple{end_time, room_id});
            m_roomsCount[room_id]++; 
        }
        return maxUsedRoom();
    }

};
void runSolX(){
    Solution sol;
    int n = 3;
    vector<vector<int>> meetings = {{39,49},{28,39},{9,29},{10,36},{22,47},{2,3},{4,49},{46,50},{45,50},{17,33}};
    int ans = sol.mostBooked(n,meetings);

}
class Solution42 {
public:

    int trap(vector<int>& height) {
        if(height.size() < 3) { return 0;}
        size_t N = height.size();
        std:vector<int> maxRight(N,0),maxLeft(N,0);
        for(size_t i = 1; i< height.size(); i++ ){
            maxLeft[i] = std::max(height[i-1], maxLeft[i-1]);
        }
        for(size_t i = height.size()-2; i >0 ; i-- ){
            maxRight[i] = std::max(maxRight[i+1], height[i+1]);
        }
    
        int left = 0,right = 0, currWater = 0, totWater= 0;
        size_t curr = 1,end = height.size()-1;
        for(size_t i = 1; i < end; i ++){
            totWater+= std::max(0, std::min( maxLeft[i], maxRight[i]) - height[i]);
        }
        return totWater;
    }   
};
void runSol42(){
    Solution42 sol;
    vector<int> height = {0,1,0,2,1,0,1,3,2,1,2,1};
    int ans = sol.trap(height);
    std::cout << "Trapped water: " << ans << std::endl;
}
int main(){
    runSol42();
    return 0;
}