#include <iostream>
#include <vector>
#include <cstdlib>
#include <memory>
#include <iomanip>
#include <math.h>
#include <Timer.hpp>

double rosenbrock(const std::vector<double>& x) {
    int n = x.size();
    double sum = 0.0;
    for (int i = 0; i < n; i += 2) {
        double term1 = x[i] * x[i] - x[i + 1];
        double term2 = x[i] - 1.0;
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    return sum;
}

void gradient(const std::vector<double>& x, std::vector<double>& grad) {
    int n = x.size();
    grad.resize(n, 0.0);
    for (int i = 0; i < n; i += 2) {
        double term1 = x[i] * x[i] - x[i + 1];
        grad[i] = 200.0 * term1 * 2 * x[i] + 2.0 * (x[i] - 1.0);
        grad[i + 1] = -200.0 * term1;
    }
}

double armijo_line_search(
    const std::vector<double>& x, 
    const std::vector<double>& grad, 
    std::vector<double>& direction,
    double c = 0.01,        // Armijo条件参数
    double alpha_init = 1.0, // 初始步长
    double rho = 0.5        // 步长缩小比例
) {
    double alpha = alpha_init;
    double f_x = rosenbrock(x);
    double grad_dot_dir = 0.0;
    for (size_t i = 0; i < grad.size(); ++i) {
        grad_dot_dir += grad[i] * direction[i];
    }

    while (true) {
        std::vector<double> x_new(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] + alpha * direction[i];
        }
        double f_x_new = rosenbrock(x_new);

        if (f_x_new <= f_x + c * alpha * grad_dot_dir) {
            return alpha;
        }

        alpha *= rho;
        if (alpha < 1e-10) {
            return alpha;
        }
    }
}

void steepest_gradient_descent(
    std::vector<double>& x, 
    double tol = 1e-7,       // 梯度范数收敛阈值
    int max_iter = 100000     // 最大迭代次数
) {
    int n = x.size();
    std::vector<double> grad(n);   
    std::vector<double> direction(n); 
    int iter = 0;
    double norm_grad; 

    do {
        gradient(x, grad);

        norm_grad = 0.0;
        for (double g : grad) {
            norm_grad += g * g;
        }
        norm_grad = std::sqrt(norm_grad);

        for (size_t i = 0; i < n; ++i) {
            direction[i] = -grad[i];
        }

        double alpha = armijo_line_search(x, grad, direction);

        for (size_t i = 0; i < n; ++i) {
            x[i] += alpha * direction[i];
        }

        iter++;
        if (iter % 100 == 0 || iter == 1) {
            std::cout << "Iter " << iter 
                      << ": f(x) = " << std::setprecision(10) << rosenbrock(x) 
                      << ", ||grad|| = " << norm_grad << std::endl;
        }

    } while (norm_grad > tol && iter < max_iter);

    std::cout << "\nConverged after " << iter << " iterations.\n";
    std::cout << "Final x: ";
    for (double val : x) {
        std::cout << std::setprecision(10) << val << " ";
    }
    std::cout << "\nFinal f(x): " << rosenbrock(x) << std::endl;
}

int main() {
    std::vector<double> x = {-1.2, 1.0};
    Timer cost;
    steepest_gradient_descent(x);
    cost.elapsed("optimize");
    return 0;
}


