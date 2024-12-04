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
double PI = 2*asin(1.0);
double calculate_scope(const double pointA,const double pointB, const int steps_dist){
    double diff = pointA - pointB;
    double det = std::sqrt(std::pow(steps_dist,2) + std::pow(diff,2));
    return std::asin(diff/det);
}
Angles calculate_window(const int i, const std::vector<double> &inputs, const int window_size){
    Angles angle_i;
    if(i<= 0){
        return angle_i;
    }
    angle_i.alpha = -1 * PI * 2;
    angle_i.beta = PI * 2;
    int j = i -1;
    double slope;
    while (j >= 0 && j > i -window_size)
    {
        slope = calculate_scope(inputs[j],inputs[i],i-j);
        if(slope > angle_i.alpha){
            angle_i.alpha = slope;
        }
        if(slope < angle_i.beta){
            angle_i.beta = slope;
        }
        j--;
    }
    return angle_i;
}
void calculate(const std::vector<double> &inputs, std::vector<Angles> &outputs,
int window)
{
    const int n_rows = inputs.size();
    for(int i = 0; i<n_rows; i ++){
        Angles angle_i = calculate_window(i,inputs,window);
        outputs.push_back(angle_i);
    }
}
int read_data(const std::string filename,std::vector<double>& inputs){
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
void write_data(const std::string filename,const std::vector<Angles>& outputs){    
    std::ofstream my_file;
    my_file.open(filename);
    for(const auto &angle_i: outputs){
        my_file << angle_i.alpha <<","<< angle_i.beta <<"," <<"\n";
    }
    my_file.close();
}
int main()
{
    int window_size = 10;
    std::vector<double> inputs(MAX_ROWS);
    const int n_rows = read_data("data/input.csv",inputs);
    if(n_rows == 0){
        return 0;
    }
    std::vector<Angles> outputs;
    outputs.reserve(inputs.size());
    calculate(inputs, outputs, window_size);
    write_data("output/window_" + std::to_string(window_size) + ".csv",outputs);
    return 0;
}