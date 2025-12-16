#include <queue>
using namespace std;

 struct TreeNode {
     int val;
     TreeNode *left;
     TreeNode *right;
     TreeNode() : val(0), left(nullptr), right(nullptr) {}
     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 };
 
class Solution230 {
public:
    int kthSmallest(TreeNode* root, int k) {
        priority_queue <int, vector<int>, greater<int> > pq;
        dfs(root,pq);
        int ans = -1;
        for(int i = 0;i<k; i++){
            ans = pq.top();
            pq.pop();
        }
        return ans;
    }
    void dfs(TreeNode* node, priority_queue <int, vector<int>, greater<int> >& pq){
        if(!node){
            return;
        }
        pq.push(node->val);
        dfs(node->left,pq);
        dfs(node->right,pq);
    }
};
class Solution394 {
public:
    void processInitStr(){
        if(m_currStr.size()  == 0){
            return;
        }
        if(m_currStrStk.size() > 0){
           m_currStr = m_currStrStk.top() +  m_currStr;
           m_currStrStk.pop();
        }
        m_currStrStk.push(m_currStr);
        m_currStr = "";
        
    }
    void processNumber(char chr){
        int n =  chr - '0';
        if ( m_lastWasNum){
            n = m_multiStk.top() * 10 + n;
            m_multiStk.pop();
        }
        m_multiStk.push(n);
    }
    string getMultiplyStr(const string& strToMultiply){
        string s;
        for(int i = 0; i < m_multiStk.top(); i++){
            s += strToMultiply;
        } 
        m_multiStk.pop(); 
        return s;
    }
    void processEndStrStack(){
        m_currStr =  m_currStrStk.top();
        m_currStrStk.pop();
        m_currStr = getMultiplyStr(m_currStr);
        if(m_multiStk.size() > 0){
            if(m_currStrStk.size() > 0){
                std::string currWord = m_currStrStk.top();
                m_currStrStk.pop();
                m_currStr = currWord + m_currStr;
            } 
            m_currStrStk.push(m_currStr);
        } else {
            m_ans+= m_currStr;
        }
        
    }
    void processEndStr(){
       
        if(m_currStr.size() != 0){
            m_currStrStk.push(m_currStr);
        } 
        processEndStrStack();
        m_currStr = "";

    } 

    string decodeString(string s) {

        for(auto chr:s){
            if(chr == '['){
                processInitStr();
                m_lastWasNum = false;
            } else if (chr >= '0' && chr <= '9') {
                processNumber(chr);
                m_lastWasNum = true;
            }else if(chr == ']'){
                processEndStr();
                m_lastWasNum = false;
            } else{
                m_currStr += chr;
                m_lastWasNum = false;
            }
        }
        if(m_currStrStk.size() > 0){
            m_ans = m_currStrStk.top() + m_ans ;
        }
        return  m_ans + m_currStr;
    }
    private:
        std::stack<string> m_currStrStk;
        std::stack<string> m_unprocStr;
        string m_currStr;
        string m_ans; 
        std::stack<int> m_multiStk;
        bool m_lastWasNum = false;

};
int main(){
    Solution230 sol;
    TreeNode* root = new TreeNode(3);
    root->left = new TreeNode(1);
    root->right = new TreeNode(4);
    root->left->right = new TreeNode(2);
    int k = 1;
    int ans = sol.kthSmallest(root,k);
    return 0;
}