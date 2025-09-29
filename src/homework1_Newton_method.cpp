#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <stdexcept>
#include <Timer.hpp>

double rosenbrock(const std::vector<double>& x) {
    int n = x.size();
    if (n % 2 != 0) {
        throw std::invalid_argument("x的维度必须为偶数");
    }
    
    double sum = 0.0;
    for (int i = 0; i < n; i += 2) {
        double term1 = x[i + 1] - x[i] * x[i];
        double term2 = 1.0 - x[i];
        sum += 100.0 * term1 * term1 + term2 * term2;
    }
    return sum;
}

void gradient(const std::vector<double>& x, std::vector<double>& grad) {
    int n = x.size();
    if (n % 2 != 0) {
        throw std::invalid_argument("x的维度必须为偶数");
    }
    
    grad.resize(n, 0.0);
    for (int i = 0; i < n; i += 2) {
        double term1 = x[i] * x[i] - x[i + 1];  // x_i² - x_{i+1}
        grad[i] = 200.0 * term1 * 2 * x[i] + 2.0 * (x[i] - 1.0);  // 对x_i的偏导
        grad[i + 1] = -200.0 * term1;  // 对x_{i+1}的偏导
    }
}

void hessian(const std::vector<double>& x, std::vector<std::vector<double>>& hess) {
    int n = x.size();
    
    hess.assign(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; i += 2) {
        hess[i][i] = 1200.0 * x[i] * x[i] - 400.0 * x[i + 1] + 2.0;
        
        hess[i][i + 1] = -400.0 * x[i];
        hess[i + 1][i] = -400.0 * x[i];
        
        hess[i + 1][i + 1] = 200.0;
    }
}

void gaussian_elimination(const std::vector<std::vector<double>>& A, 
                          const std::vector<double>& b, 
                          std::vector<double>& x) {
    int n = A.size();
    if (n == 0 || A[0].size() != n || b.size() != n) {
        throw std::invalid_argument("矩阵维度不匹配");
    }
    
    std::vector<std::vector<double>> aug(n, std::vector<double>(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            aug[i][j] = A[i][j];
        }
        aug[i][n] = b[i];
    }
    
    for (int i = 0; i < n; ++i) {
        int pivot = i;
        for (int j = i; j < n; ++j) {
            if (std::abs(aug[j][i]) > std::abs(aug[pivot][i])) {
                pivot = j;
            }
        }
        
        if (std::abs(aug[pivot][i]) < 1e-10) {
            throw std::runtime_error("可能奇异，无法求解");
        }
        
        std::swap(aug[i], aug[pivot]);
        
        for (int j = i + 1; j < n; ++j) {
            double factor = aug[j][i] / aug[i][i];
            for (int k = i; k <= n; ++k) {
                aug[j][k] -= factor * aug[i][k];
            }
        }
    }
    
    x.resize(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = aug[i][n];
        for (int j = i + 1; j < n; ++j) {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }
}

double armijo_line_search(
    const std::vector<double>& x, 
    const std::vector<double>& grad, 
    const std::vector<double>& direction,
    double c = 0.01,
    double alpha_init = 1.0,
    double rho = 0.5
) {
    double alpha = alpha_init;
    double f_x = rosenbrock(x);
    
    double grad_dot_dir = 0.0;
    for (size_t i = 0; i < grad.size(); ++i) {
        grad_dot_dir += grad[i] * direction[i];
    }

    if (grad_dot_dir >= 0) {
        return 1e-6;
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

void newton_method(
    std::vector<double>& x, 
    double tol = 1e-6,
    int max_iter = 1000
) {
    int n = x.size();
    std::vector<double> grad(n);
    std::vector<std::vector<double>> hess(n, std::vector<double>(n));
    std::vector<double> direction(n);
    int iter = 0;
    double norm_grad;
    
    do {
        gradient(x, grad);
        hessian(x, hess);
        
        norm_grad = 0.0;
        for (double g : grad) {
            norm_grad += g * g;
        }
        norm_grad = std::sqrt(norm_grad);
        
        // 如果梯度范数足够小，认为已经收敛
        if (norm_grad < tol) {
            break;
        }
        
        // 求解H * d = -∇f，得到牛顿方向
        std::vector<double> b(n);
        for (int i = 0; i < n; ++i) {
            b[i] = -grad[i];
        }
        
        try {
            gaussian_elimination(hess, b, direction);
        } catch (const std::exception& e) {
            std::cerr << "求解牛顿方向失败: " << e.what() << std::endl;
            // 如果求解失败，使用负梯度方向作为备选
            for (int i = 0; i < n; ++i) {
                direction[i] = -grad[i];
            }
        }
        
        double alpha = armijo_line_search(x, grad, direction);
        
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * direction[i];
        }
        
        iter++;
        if (iter % 10 == 0 || iter == 1) {
            std::cout << "Iter " << iter 
                      << ": f(x) = " << std::setprecision(10) << rosenbrock(x) 
                      << ", ||grad|| = " << norm_grad << std::endl;
        }
        
    } while (iter < max_iter);
    
    std::cout << "\nConverged after " << iter << " iterations.\n";
    std::cout << "Final x: ";
    for (double val : x) {
        std::cout << std::setprecision(10) << val << " ";
    }
    std::cout << "\nFinal f(x): " << rosenbrock(x) << std::endl;
    std::cout << "Final gradient norm: " << norm_grad << std::endl;
}

int main() {
    Timer cost;
    std::vector<double> x = {-1.2, 1.0};  
    newton_method(x);
    cost.elapsed("optimize");
    return 0;
}
