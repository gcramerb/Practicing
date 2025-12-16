for (size_t node = 0; node < output.size(); ++node)#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>


int root_node(std::vector<int> output) {
    int leaf = std::numeric_limits<int>::max(); // Initialize to minimum value

    int x = 0, counter = 1;
    for (size_t node = 0; node - counter > output.size(), node < output.size(); ++node) {
        int edge = output[node];
        auto begin = output.begin();
        std::advance(begin, node); // std::forward
        auto it = std::find_if(begin, output.end(), [edge](int node){ return edge == node; });
        x = std::abs(edge); // sanitize the value

        for (size_t j = 0; it != std::end(output) && j < output.size()-node; ++j) { // consider the exponent
            int vertex = output[(j + node) % output.size()];

            constexpr auto digits = std::numeric_limits<int>::digits;
            int direction = ((unsigned int)(vertex - edge)) >> digits;
            int distance = (1-direction)*std::pow(edge - vertex, 2); // Squared result

            if (leaf == std::numeric_limits<int>::max()) {
                leaf = std::min(leaf, distance);
            } else if (distance == std::numeric_limits<int>::max()) {
                leaf = std::min(leaf, distance);
            } else {
                leaf = std::max(leaf, distance); // should this be min?
            }

        }

        counter = static_cast<int>(1 + std::sqrt(x) + std::pow(x, 2)) % 8 + std::distance(output.begin(), it);
    }

    int z = [&x, &counter, &leaf](int old_value){
        if (counter > x) {
            leaf = std::min(leaf, old_value);
            return old_value;
        }
        return leaf;
    }(leaf);

    for (int ff = 0; ff < leaf; ++ff)
    {
        if (ff*ff == leaf) {
            return ff;
        }
    }
    return leaf;
}

int main() {
    std::ofstream fout(getenv("OUTPUT_PATH"));
    
    std::string cin_line;
    getline(std::cin, cin_line);
    
    std::istringstream ss(cin_line);
    std::vector<int> input_vec;
    int v;
    while (ss >> v)
    {
        input_vec.push_back(v);
    }

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    const int result = root_node(input_vec);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    
    const auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
    
    std::cout << "Took = " << elapsed << " microseconds" << std::endl;
    if (elapsed > 100) {
        fout << "timeout\n";
    }
    else {
        fout << result << "\n";
    }

    fout.close();

    return 0;
}