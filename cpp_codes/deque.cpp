#include <iostream>
#include <deque> 
using namespace std;

void printKMax(int arr[], int n, int k){
    int max_ = -1;
    int max_i;
    for(int i = 0; i< k; i ++){
        if(arr[i] > max_){
            max_ = arr[i];
            max_i = i;
        }
    }
    std::cout << arr[max_i] << " ";
    int low_rang; 
    for(int i = k; i < n ; i ++){
        low_rang = i - k +1;
        if(max_i < low_rang ){
            max_ = -1;
            for(int j = low_rang; j <= i; j ++){
                if(arr[j] > max_){
                    max_i = j;
                    max_ = arr[j];
                }
            
            }
        }
        if(arr[i] > max_){
            max_i = i;
            max_ = arr[i];
        }
        std::cout << arr[max_i] << std::endl;
    }
    //std::cout << arr[max_i] << std::endl;
}

int main(){
  
	int t;
	cin >> t;
	while(t>0) {
		int n,k;
    	cin >> n >> k;
    	int i;
    	int arr[5];
    	for(i=0;i<n;i++)
      		cin >> arr[i];
    	printKMax(arr, n, k);
    	t--;
  	}
  	return 0;
}
