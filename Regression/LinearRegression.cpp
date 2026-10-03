#include "../Tensor/Tensor.h"
class LinearRegression
{
private:
    Tensor<double> weight;
    double bias;

public: 
    LinearRegression(Tensor<double> weight, double bias)
    {

        this->weight = weight;

        this->bias = bias;
    }
};
int main() {};