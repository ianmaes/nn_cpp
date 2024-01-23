#include <cmath>
#include <initializer_list>
#include <iostream>
#include <list>
#include <memory>
#include <random>
#include <stdexcept>
#include <utility>     

template <typename T>
class Matrix
{
    public:
    int nrows, ncols;
    T* data;

    public:
    Matrix() : nrows(0), ncols(0) {}
    
    // initialize with size
    Matrix(int rows, int cols) : nrows(rows), ncols(cols)
    {
        data = new T[nrows * ncols];
    }

    // std:initializer constructor
    Matrix(int rows, int cols, const std::initializer_list<T>& list) : nrows(rows), ncols(cols)
    {
        if (list.size() != rows*cols)
        {
            throw std::invalid_argument("Initializer list does not match matrix dimensions.");
        }
        
        data = new T[nrows * ncols];

        int i = 0;
        for (const auto& value : list)
        {
            data[i++] = value;
        }
    }

    // Copy constructor
    Matrix(const Matrix& other) : 
    nrows(other.nrows), ncols(other.ncols)
    {
        data = new T[nrows * ncols];
        for (int i = 0; i < nrows * ncols; ++i) 
        {
            data[i] = other.data[i];
        }
    
    }

    // Move constructor
    Matrix(Matrix&& other): 
    nrows(other.nrows), ncols(other.ncols), data(other.data)
        {
            other.data = nullptr;
            other.nrows = 0;
            other.ncols = 0;
        };

    // Destructor
    ~Matrix() 
    {
        delete[] data;
        data = nullptr;
    }


    // OPERATORS

    // copy assignment operator
    Matrix& operator=(const Matrix& other) {
        if (this != &other) {
            delete[] data;
            nrows = other.nrows;
            ncols = other.ncols;

            data = new T[nrows * ncols];

            for (int i = 0 ; i<nrows*ncols ; i++) {
                data[i] = other.data[i];
            }
            
        }
        return *this;
    }

    // move assignment operator
    Matrix& operator=(Matrix&& other) 
    {
        if (this != &other) {
            delete[] data;

            nrows = other.nrows;
            ncols = other.ncols;
            data = std::exchange(other.data, nullptr);

            other.nrows = 0;
            other.ncols = 0;

        }
        return *this;
    }


    // Access operator
    T& operator[](const std::pair<int, int>& ij) {
        int i = ij.first;
        int j = ij.second;

        if (i<0 || i>= nrows || j<0 || j>= ncols) {
            throw std::out_of_range("Matrix index out of range");
        }

        return data[ncols * i + j];
    }


    // constant access operator
    const T& operator[](const std::pair<int, int>& ij) const {
        int i = ij.first;
        int j = ij.second;

        if (i<0 || i>= nrows || j<0 || j>= ncols) {
            throw std::out_of_range("Matrix index out of range");
        }

        return data[ncols * i + j];
    }


    // SCALAR times operator
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(U x) const {
        Matrix<typename std::common_type<T,U>::type> result(nrows, ncols);

        for (int i=0; i<nrows*ncols; i++) {
            result.data[i] = data[i] * x;
        }
        return result;
    };



    // MATRIX times operator
    template<typename U>
    Matrix<typename std::common_type<T,U>::type> operator*(const Matrix<U>& B) const {
        if (ncols != B.nrows) {
            throw std::invalid_argument("Matrices are not the same size.");
        }
        
        Matrix<typename std::common_type<T, U>::type> result(nrows, B.ncols);
        
        for (int i=0; i<nrows; i++) {
            for (int j=0; j<B.ncols;j++) {
                result[{i,j}] = 0;

                for (int k=0; k<ncols;k++) {
                    result[{i,j}] += data[ncols * i + k] * static_cast<typename std::common_type<T, U>::type>(B[{k, j}]);
                }
            }
        }
        return result;

    }



    // plus operator 
    Matrix operator+(const Matrix& other) const 
    {
        if (other.nrows != nrows || other.ncols != ncols)
        {
            throw std::invalid_argument("Matrices are not the same size!");
        }

        Matrix s(nrows, ncols);
        
        for (int i = 0; i < nrows * ncols; ++i)
        {
            s.data[i] = data[i] + other.data[i];
        }
        return s;
    }

    // min operator 
    Matrix operator-(const Matrix& other) const 
    {
        if (other.nrows != nrows || other.ncols != ncols)
        {
            throw std::invalid_argument("Matrices are not the same size!");
        }

        Matrix s(nrows, ncols);
        
        for (int i = 0; i < nrows * ncols; ++i)
        {
            s.data[i] = data[i] - other.data[i];
        }
        return s;
    }

    // Fill matrix with a value
    void fill(const T& value) {
        for (int i = 0; i < nrows * ncols; ++i) {
            data[i] = value;
        }
    }
    

    // Transpose
    Matrix transpose() const
    {
        Matrix result(ncols, nrows);

        for (int i = 0; i < nrows; ++i)
        {
            for (int j = 0; j < ncols; ++j)
            {
                result[{j, i}] = data[ncols * i + j];
            }
        }

        return result;
    }


    // get the number of rows
    int getRows() const {
        return nrows;
    }
    // get the number of columns 
    int getCols() const {
        return ncols;
    }

};

template <typename T>
class Layer {
public:

    // Pure virtual function for forward propagation
    // Takes a Matrix<T> as input and returns a Matrix<T>
    virtual Matrix<T> forward(const Matrix<T>& x) = 0;

    // Pure virtual function for backward propagation
    // Takes a Matrix<T> as input and returns a Matrix<T>
    virtual Matrix<T> backward(const Matrix<T>& dy) = 0;


    virtual ~Layer() = default;
};

template<typename T>    
class Linear : public Layer<T> {
private:
    int in_features, out_features, n_samples;
    Matrix<T> bias, weights, bias_gradients, weights_gradients, cache;

public:
    Linear(int in_features, int out_features, int n_samples, int seed)
        : in_features(in_features), out_features(out_features), n_samples(n_samples),
          bias(1, out_features), weights(in_features, out_features),
          bias_gradients(1, out_features), weights_gradients(in_features, out_features),
          cache(n_samples, in_features) {
        
        std::default_random_engine generator(seed);
        std::normal_distribution<T> distribution_normal(0.0, 1.0);
        std::uniform_real_distribution<T> distribution_uniform(0.0, 1.0);

        // Initialize weights and bias
        for (int i = 0; i < in_features; ++i) {
            for (int j = 0; j < out_features; ++j) {
                weights[{i, j}] = distribution_normal(generator);
            }
        }
        for (int j = 0; j < out_features; ++j) {
            bias[{0, j}] = distribution_uniform(generator);
        }

        // Initialize gradients to zero
        // Assume Matrix class has method to fill with zeros
        bias_gradients.fill(0);
        weights_gradients.fill(0);
    }

    virtual ~Linear() {}

    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        cache = x; // Storing x in cache for use in backward pass
        return (x * weights) + bias; // y = x * w + b
    }

    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        // Calculating gradients
        for (int i = 0; i < n_samples; ++i) {
            for (int j = 0; j < out_features; ++j) {
                bias_gradients.data[j] += dy[{i, j}];
        }
    }
        weights_gradients = cache.transpose() * dy;

        // Calculating downstream gradient
        return dy * weights.transpose(); // dL/dx = dL/dy * w^T
    }

    void optimize(T learning_rate) {
        // Update weights and bias using gradients
        weights = weights - (weights_gradients * learning_rate);
        bias = bias - (bias_gradients * learning_rate);
    }
};

template<typename T>
class ReLU : public Layer<T> {
private:
    int in_features, out_features, n_samples;
    Matrix<T> cache;

public:
    ReLU(int in_features, int out_features, int n_samples)
        : in_features(in_features), out_features(out_features), n_samples(n_samples),
          cache(n_samples, in_features) {
        // Ensure in_features equals out_features for ReLU layer
        if (in_features != out_features) {
            throw std::invalid_argument("In_features and out_features must be equal for ReLU layers.");
        }
    }

    virtual ~ReLU() {}

    virtual Matrix<T> forward(const Matrix<T>& x) override final {
        cache = x;
        Matrix<T> output(x.getRows(), x.getCols());

        for (int i = 0; i < x.getRows(); ++i) {
            for (int j = 0; j < x.getCols(); ++j) {
                T value = x[{i, j}];
                output[{i, j}] = (value > 0) ? value : 0;
            }
        }
        return output;
    }

    virtual Matrix<T> backward(const Matrix<T>& dy) override final {
        Matrix<T> dReLU_dy(cache.getRows(), cache.getCols());

        for (int i = 0; i < cache.getRows(); ++i) {
            for (int j = 0; j < cache.getCols(); ++j) {
                dReLU_dy[{i, j}] = (cache[{i, j}] > 0) ? 1 : 0;
            }
        }

        // Element-wise multiplication of dy and dReLU_dy
        Matrix<T> downstream_gradient = dy * dReLU_dy; 
        return downstream_gradient;
    }
};


template<typename T>
class Net {
private:
    // Using smart pointers for automatic memory management
    Layer<T>* layer1;
    Layer<T>* layer2;
    Layer<T>* layer3;


public:
    // Constructor
    Net(int in_features, int hidden_dim, int out_features, int n_samples, int seed) {
        // Initialize the layers
        layer1 = std::make_unique<Linear<T>>(in_features, hidden_dim, n_samples, seed);
        layer2 = std::make_unique<ReLU<T>>(hidden_dim, hidden_dim, n_samples);
        layer3 = std::make_unique<Linear<T>>(hidden_dim, out_features, n_samples, seed);
    }

    // Destructor - default is fine since we're using smart pointers
    ~Net() {
        delete layer1;
        delete layer2;
        delete layer3;
    }

    // Forward function
    Matrix<T> forward(const Matrix<T>& x) {
        auto output1 = layer1->forward(x);
        auto output2 = layer2->forward(output1);
        auto output3 = layer3->forward(output2);
        return output3;
    }

    // Backward function
    Matrix<T> backward(const Matrix<T>& dy) {
        auto grad1 = layer3->backward(dy);
        auto grad2 = layer2->backward(grad1);
        auto grad3 = layer1->backward(grad2);
        return grad3;
    }

    // Optimize function
    void optimize(T learning_rate) {
        // Assuming Linear layers have an optimize method
        // dynamic_cast is used to ensure the layer is a Linear layer
        if (auto linearLayer = dynamic_cast<Linear<T>*>(layer1.get())) {
            linearLayer->optimize(learning_rate);
        }
        if (auto linearLayer = dynamic_cast<Linear<T>*>(layer3.get())) {
            linearLayer->optimize(learning_rate);
        }
    }


};

// Function to calculate the loss
template <typename T>
T MSEloss(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
    // Your implementation of the MSEloss function starts here
};

// Function to calculate the gradients of the loss
template <typename T>
Matrix<T> MSEgrad(const Matrix<T>& y_true, const Matrix<T>& y_pred) 
{
    // Your implementation of the MSEgrad function starts here
}

// Calculate the argmax 
template <typename T>
Matrix<T> argmax(const Matrix<T>& y) 
{
    // Your implementation of the argmax function starts here
}

// Calculate the accuracy of the prediction, using the argmax
template <typename T>
T get_accuracy(const Matrix<T>& y_true, const Matrix<T>& y_pred)
{
    // Your implementation of the get_accuracy starts here
}

int main(int argc, char* argv[])
{
    Matrix A(2, 2, {1., 0., 1., 1.});
    Matrix B(2, 2, {2., 0., 0., 4.});
    

    Matrix C = A*B;
    for(int i = 0; i < 4; ++i)
    {
        std::cout << C.data[i] << " ";
    }
    return 0;
}