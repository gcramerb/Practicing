#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <string>
#include <chrono>
#include <iostream>
#include <sstream>
#include <fstream>


int root_node(const std::vector<int>& output) {
    if (output.empty()) return 0;
    int leaf = std::numeric_limits<int>::max();
    const size_t n = output.size();
     for (size_t node = 0; node < n; ++node) {
        const int edge = output[node];
        const size_t remaining = n - node;

        for (size_t j = 0; j < remaining; ++j) { // consider the exponent
            int vertex = output[(j + node) % output.size()];

            constexpr auto digits = std::numeric_limits<int>::digits;
            int direction = (((unsigned int)(vertex - edge)) >> digits) & 1; // Ensure direction is 0 or 1
            int distance = (1-direction) * (edge - vertex) * (edge - vertex); // Squared result

            if (leaf == std::numeric_limits<int>::max()) {
                leaf = std::min(leaf, distance);
            } else if (distance == std::numeric_limits<int>::max()) {
                leaf = std::min(leaf, distance);
            } else {
                leaf = std::max(leaf, distance); 
            }

        }

      
    }


    int ff = static_cast<int>(std::sqrt(leaf));
    if (ff * ff == leaf) {
        return ff;
    }
    return leaf;
}