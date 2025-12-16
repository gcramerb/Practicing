#include <bits/stdc++.h>
#include <vector>

using namespace std;

string ltrim(const string &);
string rtrim(const string &);

/** `perform_op` applies the specified arithmetic `op` to
 *  `lhs` and `rhs`. */
long perform_op( char op, long lhs, long rhs )
{
  switch (op)
  {
    case '+': return lhs + rhs;
    case '-': return lhs - rhs;
    case '*': return lhs * rhs;
    case '/': return lhs / rhs;
  }
  throw std::runtime_error{ std::string{"Bad op: "} + op };
}

/*
 operators: + - * / 
 input -> +**2,11,-1,4,*2,3
 
 stack[+]
 stack_num [2,3]


 
curr_ans <- 2*11
curr_ans <- curr_ans * (1-4)
curr_ans <- curr_ans * (2*3)


input -> * / 2 7 1
 stack[ ]
 stack_num []
curr_ans = 2 / 7

curr_ans *= 1

input -> + + + * / 2 7 + 1 1 + 1 1 + 1 1 1 1
 stack[+ ]
 stack_num [ 1]
 
 bool curr_ans_init = false
 bool operation_just_done = false
 
 curr_ans = 2/7 
 
tmp  = 1 + 1
curr_ans * tmp
operation_just_done = false
tmp  = 1 + 1
curr_ans + tmp


operation_just_done = true
tmp  = 1 + 1
curr_ans + tmp
--------------------------




*/

long perform_calculation( std::string input ) {
    
    bool curr_ans_init = false;
    bool operation_just_done = false;
    long ans = 0;
    long tmp,num;
    long last_num = -1;
    std::vector<char> ops;
    for(const auto&c : input){
        if(c == ',') continue;
        if(c == '*' || c == '/' || c == '+' || c == '-'){
            ops.push_back(c);
            operation_just_done = false;
            last_num = -1;
        }else{
            num = long(c);
            if(operation_just_done) {
                ans  = perform_op(ops.back(),ans,num);
                ops.pop_back();
            } else if (last_num > 0){
                tmp  = perform_op(ops.back(),last_num,num);
                ops.pop_back();
                if (curr_ans_init){
                    ans  = perform_op(ops.back(),ans,tmp);
                    ops.pop_back();
                }else {
                    ans = tmp;
                }
                operation_just_done = true;
            }else {
                last_num = num;
            }
            
        } 
    }

    return ans;

}

vector<long> perform_calculations( vector<string> inputCalculations )
{
  vector<long> result;
  result.reserve( inputCalculations.size() );
  std::transform( inputCalculations.begin(),
                  inputCalculations.end(),
                  std::back_inserter( result ),
                  &perform_calculation );
  return result;
}

int main()
{
    ofstream fout(getenv("OUTPUT_PATH"));

    string inputs_count_temp;
    getline(cin, inputs_count_temp);

    int inputs_count = stoi(ltrim(rtrim(inputs_count_temp)));

    vector<string> inputs(inputs_count);

    for (int i = 0; i < inputs_count; i++) {
        string inputs_item;
        getline(cin, inputs_item);

        inputs[i] = inputs_item;
    }

    vector<long> result = perform_calculations(inputs);

    for (size_t i = 0; i < result.size(); i++) {
        fout << result[i];

        if (i != result.size() - 1) {
            fout << "\n";
        }
    }

    fout << "\n";

    fout.close();

    return 0;
}

string ltrim(const string &str) {
    string s(str);

    s.erase(
        s.begin(),
        find_if(s.begin(), s.end(), not1(ptr_fun<int, int>(isspace)))
    );

    return s;
}

string rtrim(const string &str) {
    string s(str);

    s.erase(
        find_if(s.rbegin(), s.rend(), not1(ptr_fun<int, int>(isspace))).base(),
        s.end()
    );

    return s;
}