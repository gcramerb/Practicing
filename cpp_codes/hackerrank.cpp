#include <cmath>
#include <cstdio>
#include <vector>
#include <map>
#include <iostream>
#include <sstream>
#include <algorithm>
using namespace std;


void addMappingAttribute(map<string,string>& _map, string& key, const vector<string>& tag,const string& data){
    if(data.size() == 0){
        return;
    }
    if(key.size() == 0){
        key = data;
    }else{
        _map[*tag.rbegin()+"~"+key] = data;
        //cout <<"adding: " << tag + "~" +key << " -> " <<data << endl;
        key.clear();
    }
}
int main() {
    int N,Q;
    cin >> N >> Q;
    cin.ignore();
    std::map<string,string> _map;
    vector<string> tag;
    string key;
    for(int i = 0; i< N; i ++){
        string line, curr_tag, data;
        getline(cin,line);
        line.erase(remove(line.begin(), line.end(), '\"' ),line.end());
        line.erase(remove(line.begin(), line.end(), '>' ),line.end());
        line.erase(remove(line.begin(), line.end(), '=' ),line.end());
        stringstream ss(line);
        while (getline(ss,data,' ')) {
            auto found = data.find('/');
            if (found != std::string::npos){
                tag.pop_back();
                continue;
            }
            if(data[0] == '<'){
                curr_tag = data.substr(1);
                string temp1="";
                if(tag.size()>){
                    temp1=*tag.rbegin();
                    temp1=temp1+"."+curr_tag;
                }else{
                    temp1=curr_tag;
                }
                tag.push_back(temp1);
            }else {
                addMappingAttribute(_map,key,tag,data);
            }
        }

    }

    for(int i = 0; i< Q; i ++){
        string line;
        getline(cin,line);
        //cout << "Quering: " <<line <<endl;
        if (_map.find(line) == _map.end()) {
            cout << "Not Found!" <<std::endl;
        } else {
            cout << _map[line] <<std::endl;
        }
    }
    return 0;
}






#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;


int main() {
    /* Enter your code here. Read input from STDIN. Print output to STDOUT */  
    int[10] nums; 
    unsigned int N, a ,b, x;
     std::vector<int> vec;
     std::cin >> N;
     for(int i = 0; i< N; i++){
        std::cin >> x;
        vec.push_back(x);
     }
    std::cin >> N;
    std::cin >> a >> b;

    return 0;
}

