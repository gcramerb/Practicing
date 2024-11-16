
#include <string> 
#include <vector> 
#include <iostream>
#include <list>
#include <limits>
using namespace std;

class Solution {
public:
    int findMinDifference(vector<string>& timePoints) {
        int N = timePoints.size();
        int diff = std::numeric_limits<int>::max();
        int diff_i;
        for(int i = 0; i< N; i++){
            for(int j = i +1; j <N; j++){
                diff_i = getDiff(timePoints[i],timePoints[j]);
                if(diff_i < diff){
                    diff = diff_i;
                }
                if(diff == 0) return diff;
            }
        }
        return diff;
    }
    int getHourRightDay(int hourA,int hourB){
        if(hourA > 12 && hourB < 12){
            if(std::abs(hourB+24 - hourA) < std::abs(hourB - hourA)){
                hourB+=24;
            }
        }
        return hourB;
    }
    int getDiff(string a, string b){
        int  hourA = std::stoi(a.substr(0, a.find(":"))); 
        int  hourB = std::stoi(b.substr(0, b.find(":")));
        a.erase(0, a.find(":") + 1);
        b.erase(0, b.find(":") + 1);
        int  minA = std::stoi(a);
        int  minB = std::stoi(b);
        hourA = getHourRightDay(hourB,hourA);
        hourB = getHourRightDay(hourA,hourB);
        minA = hourA*60 + minA;
        minB = hourB*60 + minB;
        return std::abs(minA- minB);
    }
};

int main(){
    vector<string> testCase1 = {"00:00","23:59","00:00"};
    vector<string> testCase2 = {"01:01","02:02"};
    vector<string> testCase3 = {"01:01","02:01","03:00","03:01"};
    vector<string> testCase4 = {"02:39","10:26","21:43"};
    Solution s;
    int awns = s.findMinDifference(testCase4);
    cout << awns;
}