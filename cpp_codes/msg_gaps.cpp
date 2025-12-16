#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <utility>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;

/*

message -> ID unsigned int; string message


num_max gaps = unsigned int = MAX_GAPS; # eg: 10; 

100, 200, 201, 202, 203,....,500,501, 502, ..., 600, 700, 101, 102, 701,702,703,704 -> [{101,199}, {203,500}, {502,600} , {600,700}]
inp = 200; - > 

inp <= lasID ? 
    return true; # is duplicated 
return false; 

1,2,3, 1000, 1001, 200,201, 202 , -> possible  -> [ 4,999]  

1,2,3,5,6,7,4, -> NO_GAPS -> [7] 


duplicated -> ignore do not prcess the message! 



*/
MAX_GAPS = 10;
class SeqGapGurad
{
    public:
    bool is_dup(std::uint64_t seq_no)
    {
        for( const std::pair<std::uint64_t, std::uint64_t>& gap:gaps){
            if(seq_no >  gap.first && seq_no < gap.second){
                return false;
            }
            if(seq_no <= lastID){
                return true;
            }
            return false;
        }
        
    }
    void  get_num(std::uint64_t seq_no){
        if (seq_no == lastID +1 ){
            lastID = seq_no;
            return;
        }
        
        for(int i = 0; i < MAX_GAPS; i ++){
            // update 
            if(gaps[i].first == seq_no){
                gaps[i].first == seq_no +1;
                
            }
            if(gaps[i].second == seq_no){
                gaps[i].first == seq_no -1;
                
            }
            //split: 
            if(seq_no >  gaps[i].first && seq_no < gaps[i].second){
                
            }
            

            if(gaps[i].first == gaps[i].second){ // removing the gap
                gaps[i].first  = 0;
                gaps[i].second = 0;
            }
            
            // gap creation: 
            
            
        }
    }
    private:
        std::uint64_t lastID; 
        std::array<std::pair<std::uint64_t, std::uint64_t> , MAX_GAPS> gaps; // assume that I will keep it sorted! TODO! 
};