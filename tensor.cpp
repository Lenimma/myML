#include <iostream>
#include <vector>
#include <random>
#include <initializer_list>
#include <stdexcept>
#include <type_traits>

using namespace std;

template <typename T>
class Tensor
{
private:
    vector<T> data;
    vector<int> shape;
    vector<int> strides;
    int data_size;

    int data_sz(const vector<int>& shape)
    {
        int output = 1;
        for (auto s : shape)
        {
            if (s < 0)
                throw invalid_argument("shape dimensions must be non-negative");

            output *= s;
        }
        return output;
    }

    int data_sz(initializer_list<int> shape)
    {
        int output = 1;
        for (auto s : shape)
        {
            if (s < 0)
                throw invalid_argument("shape dimensions must be non-negative");

            output *= s;
        }
        return output;
    }

    void calculate_strides()
    {
        strides.resize(shape.size());

        int sl = 1;

        for (int i = static_cast<int>(shape.size()) - 1; i >= 0; i--)
        {
            strides[i] = sl;
            sl *= shape[i];
        }
    }

    int get_data_index(initializer_list<int> list) const
    {
        if (list.size() != shape.size())
            throw invalid_argument("number of indices does not match tensor dimensions");

        int data_index = 0;
        int i = 0;

        for (auto index : list)
        {
            if (index < 0 || index >= shape[i])
                throw invalid_argument("index out of bounds");

            data_index += index * strides[i];
            i++;
        }

        return data_index;
    }

    void check_same_shape(const Tensor& other) const
    {
        if (shape != other.shape)
            throw invalid_argument("tensor shapes do not match");
    }

public:
    Tensor()
        : data_size(0)
    {
    }

    Tensor(int n)
    {
        if (n < 0)
            throw invalid_argument("size must be non-negative");

        data_size = n;
        data = vector<T>(n);
        shape = {n};
        calculate_strides();
    }

    Tensor(initializer_list<int> list)
    {
        shape = vector<int>(list);
        data_size = data_sz(list);
        data = vector<T>(data_size);
        calculate_strides();
    }

    int size() const
    {
        return static_cast<int>(data.size());
    }

    int shape_i(int index) const
    {
        if (index < 0 || index >= static_cast<int>(shape.size()))
            throw invalid_argument("shape index out of bounds");

        return shape[index];
    }

    int strides_i(int index) const
    {
        if (index < 0 || index >= static_cast<int>(strides.size()))
            throw invalid_argument("stride index out of bounds");

        return strides[index];
    }

    int ndim() const
    {
        return static_cast<int>(shape.size());
    }

    void asign(initializer_list<int> list, T input)
    {
        int data_index = get_data_index(list);
        data[data_index] = input;
    }

    void flat_asign(int index, T value)
    {
        if (index < 0 || index >= static_cast<int>(data.size()))
            throw invalid_argument("flat asign error");

        data[index] = value;
    }

    T& at(initializer_list<int> list)
    {
        return data[get_data_index(list)];
    }

    const T& at(initializer_list<int> list) const
    {
        return data[get_data_index(list)];
    }

    T& operator[](initializer_list<int> list)
    {
        return data[get_data_index(list)];
    }

    const T& operator[](initializer_list<int> list) const
    {
        return data[get_data_index(list)];
    }

    T& operator[](int index)
    {
        if (index < 0 || index >= static_cast<int>(data.size()))
            throw invalid_argument("index out of bounds");

        return data[index];
    }

    const T& operator[](int index) const
    {
        if (index < 0 || index >= static_cast<int>(data.size()))
            throw invalid_argument("index out of bounds");

        return data[index];
    }

    static Tensor arrange(int n)
    {
        Tensor output(n);

        for (int i = 0; i < n; i++)
        {
            output.flat_asign(i, static_cast<T>(i));
        }

        return output;
    }

    static Tensor arange(initializer_list<int> list)
    {
        Tensor output(list);

        for (int i = 0; i < output.size(); i++)
        {
            output.flat_asign(i, static_cast<T>(i));
        }

        return output;
    }

    static Tensor zeros(int n)
    {
        return Tensor(n);
    }

    static Tensor zeros(initializer_list<int> list)
    {
        return Tensor(list);
    }

    static Tensor full(initializer_list<int> list, T input)
    {
        Tensor output(list);

        for (int i = 0; i < output.size(); i++)
        {
            output.flat_asign(i, input);
        }

        return output;
    }

    static Tensor ones(initializer_list<int> list)
    {
        return full(list, static_cast<T>(1));
    }

    static Tensor uniform(initializer_list<int> dimensions, T start, T end)
    {
        static_assert(
            is_floating_point<T>::value,
            "uniform requires a floating point Tensor type"
        );

        if (start > end)
            throw invalid_argument("start must be less than or equal to end");

        uniform_real_distribution<T> distribution(start, end);
        random_device rd;
        mt19937 gen(rd());

        Tensor output(dimensions);

        for (int i = 0; i < output.size(); i++)
        {
            output.flat_asign(i, distribution(gen));
        }

        return output;
    }

    static Tensor normal(
        initializer_list<int> dimensions,
        double mean,
        double deviation)
    {
        static_assert(
            is_floating_point<T>::value,
            "normal requires a floating point Tensor type"
        );

        if (deviation < 0)
            throw invalid_argument("standard deviation must be non-negative");

        normal_distribution<T> distribution(
            static_cast<T>(mean),
            static_cast<T>(deviation)
        );

        random_device rd;
        mt19937 gen(rd());

        Tensor output(dimensions);

        for (int i = 0; i < output.size(); i++)
        {
            output.flat_asign(i, distribution(gen));
        }

        return output;
    }

    static Tensor standard(initializer_list<int> dimensions)
    {
        return normal(dimensions, 0.0, 1.0);
    }

    Tensor operator+(T input) const
    {
        Tensor output(*this);

        for (auto& value : output.data)
        {
            value += input;
        }

        return output;
    }

    Tensor& operator+=(T input)
    {
        for (auto& value : data)
        {
            value += input;
        }

        return *this;
    }

    Tensor operator-(T input) const
    {
        Tensor output(*this);

        for (auto& value : output.data)
        {
            value -= input;
        }

        return output;
    }

    Tensor& operator-=(T input)
    {
        for (auto& value : data)
        {
            value -= input;
        }

        return *this;
    }

    Tensor operator*(T input) const
    {
        Tensor output(*this);

        for (auto& value : output.data)
        {
            value *= input;
        }

        return output;
    }

    Tensor& operator*=(T input)
    {
        for (auto& value : data)
        {
            value *= input;
        }

        return *this;
    }

    Tensor operator/(T input) const
    {
        if (input == static_cast<T>(0))
            throw invalid_argument("division by zero");

        Tensor output(*this);

        for (auto& value : output.data)
        {
            value /= input;
        }

        return output;
    }

    Tensor& operator/=(T input)
    {
        if (input == static_cast<T>(0))
            throw invalid_argument("division by zero");

        for (auto& value : data)
        {
            value /= input;
        }

        return *this;
    }

    Tensor& operator=(T input)
    {
        for (auto& value : data)
        {
            value = input;
        }

        return *this;
    }

    Tensor operator+(const Tensor& other) const
    {
        check_same_shape(other);

        Tensor output(shape);

        for (int i = 0; i < data_size; i++)
        {
            output.data[i] = data[i] + other.data[i];
        }

        return output;
    }

    Tensor operator-(const Tensor& other) const
    {
        check_same_shape(other);

        Tensor output(shape);

        for (int i = 0; i < data_size; i++)
        {
            output.data[i] = data[i] - other.data[i];
        }

        return output;
    }

    Tensor operator*(const Tensor& other) const
    {
        check_same_shape(other);

        Tensor output(shape);

        for (int i = 0; i < data_size; i++)
        {
            output.data[i] = data[i] * other.data[i];
        }

        return output;
    }

    Tensor operator/(const Tensor& other) const
    {
        check_same_shape(other);

        Tensor output(shape);

        for (int i = 0; i < data_size; i++)
        {
            if (other.data[i] == static_cast<T>(0))
                throw invalid_argument("division by zero");

            output.data[i] = data[i] / other.data[i];
        }

        return output;
    }

    Tensor& operator+=(const Tensor& other)
    {
        check_same_shape(other);

        for (int i = 0; i < data_size; i++)
        {
            data[i] += other.data[i];
        }

        return *this;
    }

    Tensor& operator-=(const Tensor& other)
    {
        check_same_shape(other);

        for (int i = 0; i < data_size; i++)
        {
            data[i] -= other.data[i];
        }

        return *this;
    }

    Tensor& operator*=(const Tensor& other)
    {
        check_same_shape(other);

        for (int i = 0; i < data_size; i++)
        {
            data[i] *= other.data[i];
        }

        return *this;
    }

    Tensor& operator/=(const Tensor& other)
    {
        check_same_shape(other);

        for (int i = 0; i < data_size; i++)
        {
            if (other.data[i] == static_cast<T>(0))
                throw invalid_argument("division by zero");

            data[i] /= other.data[i];
        }

        return *this;
    }

    Tensor transpose() const
    {
        if (ndim() != 2)
            throw invalid_argument("transpose currently requires a 2D tensor");

        Tensor output({shape[1], shape[0]});

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < shape[1]; j++)
            {
                output[{j, i}] = data[i * strides[0] + j * strides[1]];
            }
        }

        return output;
    }

    Tensor matmul(const Tensor& other) const
    {
        if (ndim() != 2 || other.ndim() != 2)
            throw invalid_argument("matmul requires 2D tensors");

        if (shape[1] != other.shape[0])
            throw invalid_argument("matrix dimensions do not match");

        Tensor output({shape[0], other.shape[1]});

        for (int i = 0; i < shape[0]; i++)
        {
            for (int j = 0; j < other.shape[1]; j++)
            {
                T sum = static_cast<T>(0);

                for (int k = 0; k < shape[1]; k++)
                {
                    sum +=
                        data[i * strides[0] + k * strides[1]] *
                        other.data[k * other.strides[0] + j * other.strides[1]];
                }

                output[{i, j}] = sum;
            }
        }

        return output;
    }

    T sum() const
    {
        T output = static_cast<T>(0);

        for (const auto& value : data)
        {
            output += value;
        }

        return output;
    }

    T mean() const
    {
        if (data.empty())
            throw invalid_argument("cannot calculate mean of empty tensor");

        return sum() / static_cast<T>(data.size());
    }

    void print_recursive(int dimension, int offset) const
    {
        if (dimension == ndim())
        {
            cout << data[offset];
            return;
        }

        cout << "[";

        for (int i = 0; i < shape[dimension]; i++)
        {
            if (i > 0)
                cout << ", ";

            print_recursive(
                dimension + 1,
                offset + i * strides[dimension]
            );
        }

        cout << "]";
    }

    void print() const
    {
        print_recursive(0, 0);
        cout << '\n';
    }

    void print_shape() const
    {
        cout << "[";

        for (size_t i = 0; i < shape.size(); i++)
        {
            cout << shape[i];

            if (i + 1 < shape.size())
                cout << ",";
        }

        cout << "]\n";
    }

    void reshape(initializer_list<int> list)
    {
        int sz = data_sz(list);

        if (sz != data_size)
            throw invalid_argument("new shape must contain the same number of elements");

        shape = vector<int>(list);
        calculate_strides();
    }

    bool operator==(const Tensor& other) const
    {
        if (shape != other.shape)
            return false;

        if (data_size != other.data_size)
            return false;

        for (int i = 0; i < data_size; i++)
        {
            if (data[i] != other.data[i])
                return false;
        }

        return true;
    }

    bool operator!=(const Tensor& other) const
    {
        return !(*this == other);
    }
};

int main()
{
    Tensor<double> a = Tensor<double>::ones({3, 3});
    Tensor<double> b = Tensor<double>::zeros({3, 3});

    int c = a == b;

    cout << c << '\n';

    Tensor<double> x = Tensor<double>::arange({2, 3});
    x.print();

    Tensor<double> y = x.transpose();
    y.print();

    Tensor<double> z = x.matmul(y);
    z.print();

    cout << z.mean() << '\n';

    Tensor<double> r = Tensor<double>::normal({2, 3}, 0, 1);
    r.print();
}