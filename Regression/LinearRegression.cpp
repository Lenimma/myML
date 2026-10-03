#include "../Tensor/Tensor.h"
class LinearRegression
{
private:
    Tensor<double> weight;
    double bias;
    double lr;
    char GD_type;

public:
    LinearRegression(int numbeer_of_features, double lr, char GD_type)
    {
        weight = Tensor<double>::zeros({numbeer_of_features, 1});

        bias = 0;

        this->lr = lr;

        this->GD_type = GD_type;
    }
    LinearRegression(Tensor<double> weight, double bias, double lr, char GD_type)
    {

        this->weight = weight;

        this->bias = bias;

        this->lr = lr;

        this->GD_type = GD_type;
    }

    void fit(Tensor<double> &X, Tensor<double> y)
    {
        if (X.shape_i(0) != y.shape_i(0))
        {
            throw std::invalid_argument("Number of samples in X and y must be the same.");
        }
        if (X.shape_i(1) != weight.shape_i(0))
        {
            throw std::invalid_argument("Number of features in X must match the number of weights.");
        }
        int n_samples = X.shape_i(0);
        int batch_size;

        if (GD_type == 'B')
            batch_size = n_samples;
        else if (GD_type == 'S')
            batch_size = 1;
        else if (GD_type == 'M')
            batch_size = 64;

        int start = 0, end = batch_size;
        while (end < n_samples)
        {
            Tensor<double> y_pred = predict(X.slice(start,end,0));
            Tensor<double> error = y_pred - y.slice(start,end,0);

            weight = weight - (X.slice(start,end,0).transpose().matmul(error) * lr) / (start-end);

            bias = bias - (error.sum() * lr) / (start-end);

            start=end;
            end=min(end+batch_size,n_samples);
        }
    }
    const Tensor<double> predict(Tensor<double> X)
    {
        return X.matmul(weight) + bias;
    }
    void set_lr(double lr)
    {
        this->lr = lr;
    }
    void set_GD_type(char GD_type)
    {
        this->GD_type = GD_type;
    }
};