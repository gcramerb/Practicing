#include <iostream>
#include <vector>
#include <map>
#include <string>
#include <algorithm>
#include <set>
#include <cassert>
using namespace std;

struct Node{
   Node* next;
   Node* prev;
   int value;
   int key;
   Node(Node* p, Node* n, int k, int val):prev(p),next(n),key(k),value(val){};
   Node(int k, int val):prev(NULL),next(NULL),key(k),value(val){};
};

class Cache{
   
   protected: 
   map<int,Node*> mp; //map the key to the node in the linked list
   int cp;  //capacity
   Node* tail; // double linked list tail pointer
   Node* head; // double linked list head pointer
   virtual void set(int, int) = 0; //set function
   virtual int get(int) = 0; //get function

};


class LRUCache: public Cache {
  public:
    LRUCache(int c){
        cp = c;
        tail = nullptr;
        head = nullptr;
    } 
    void set_key_head(const int key, const int value){
        Node* new_head = mp[key];
        new_head->value = value;
        if(!new_head->next){ // already is the head
            return;
        }
        new_head->next->prev = new_head->prev;
        if(new_head->prev){
            new_head->prev->next = new_head->next;
        }else{
            tail = new_head->next;
        }
        new_head->prev = head;
        head = new_head;
        new_head->next = nullptr;
    }
    void delete_tail(){
        if(!tail){
            return;
        }
        int tail_key = tail->key;
        Node* tmp = tail->next;
        delete tail;
        tail = tmp;
        mp.erase(tail_key);
    }
    void set_key_new_ele(const int key, const int value){
        Node* new_head = new Node(key,value);
        new_head->prev = head;
        head = new_head; 
        
        if(cp > 0){
            cp--;
            if(!tail){
                tail = new_head;
                tail->prev = nullptr;
                tail->next = new_head->next;
            }
        }else{
            delete_tail();
        }
        mp.insert({key,new_head});
    }
    void set(int key, int value) override{
        if (mp.find(key) != mp.end()) {
            set_key_head( key, value);
        }else{
            set_key_new_ele( key, value);
        }
    } 

    int get(int key) override {
        if (mp.find(key) != mp.end()) {
            return mp[key]->value;
        }
        return -1;
    } 
           
};

int main() {
   int n, capacity,i;
   n = 8;
   //cin >> n >> capacity;
   capacity = 4;
   std::vector<std::string> cmds = {"set","set","get","set","set","set","get","get"};
   std::vector<int> gets = {-1,-1,2,-1,-1,-1,4,5};
   std::vector<std::pair<int,int>> sets = {{4,2},{2,7},{-1,-1},{1,8},{5,9},{6,15}};
   LRUCache l(capacity);
   for(i=0;i<n;i++) {
      string command = cmds[i];
      //cin >> command;
      if(command == "get") {
         int key = gets[i];
         cout << l.get(key) << endl;
      } 
      else if(command == "set") {
         int key, value;
         key = sets[i].first;
         value = sets[i].second;
         l.set(key,value);
      }
   }
   return 0;
}
