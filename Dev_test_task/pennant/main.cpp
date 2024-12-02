#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <format>
#include <math.h>
const int MAX_ROWS = 100000;
struct Angles {
double alpha{0};
double beta{0};
};

double calculate_scope(const double pointA,const double pointB, const int steps_dist){
    double diff = pointA - pointB;
    double det = std::sqrt(std::pow(steps_dist,2) + std::pow(diff,2));
    return std::asin(diff/det);
}
void calculate(const std::vector<double> &inputs, std::vector<Angles> &result,
int window)
{
    const int n_rows = inputs.size();
    int j;
    double pi = 2*asin(1.0);
    for(int i = 1; i<n_rows; i ++){
        j = i -1;
        double alpha = -1 * pi * 2;
        double beta = pi * 2;
        while (j >= 0 && j > i -window)
        {
            double slope = calculate_scope(inputs[j],inputs[i],i-j);
            if(slope > alpha){
                alpha = slope;
            }
            if(slope < beta){
                beta = slope;
            }
            j--;
        }
        Angles angle_i;
        angle_i.alpha = alpha;
        angle_i.beta = beta;
        result[i] = angle_i;
    }
}
int read_data(std::string filename,std::vector<double>& inputs){
    std::string line;
    int row = 0;
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file!" << std::endl;
        return 0;
    }
    while (getline(file, line) && row < MAX_ROWS) {
        std::stringstream ss(line);
        std::string cell;
        getline(ss, cell, ',');
        inputs[row] = std::stod(cell);
        row ++;
    }
    return row;
}
void write_data(std::string filename,const std::vector<Angles>& outputs){    
    std::ofstream my_file;
    my_file.open(filename);
    Angles a_i;
    const int n_rows = outputs.size();
    for(int i = 0; i<n_rows; i ++){
        a_i = outputs[i];
        my_file << a_i.alpha <<","<< a_i.beta <<"," <<"\n";
    }
    my_file.close();
}
int main()
{
    int window_size = 10;
    std::vector<double> inputs(MAX_ROWS);
    const int n_rows = read_data("data/input.csv",inputs);
    std::vector<Angles> outputs;
    outputs.reserve(inputs.size());
    calculate(inputs, outputs, window_size);
    write_data("output/window_" + std::to_string(window_size) + ".csv",outputs);
    return 0;
}