
# In this exercise you will implement a regularized polynomial regression.
# You will need some basic numerical linear algebra to solve this problem,
# it can be solved in three lines of code.
#
# Hint: you rephrase it as a system of linear equations and solve it with
# numpys lstsq-function.
#
import numpy as np
import matplotlib.pyplot as plt

# fit a polynomial of degree k to data (x,y) with regularization parameter lamb
def get_reg_poly_model(x, y, k, alpha=0):
    # Build Vandermonde matrix for polynomial features
    X = np.vander(x, k+1, increasing=False)
    # Regularization matrix (ridge regression)
    reg = alpha * np.eye(k+1)
    # Solve (X^T X + lambda*I) w = X^T y
    w = np.linalg.lstsq(X.T @ X + reg, X.T @ y, rcond=None)[0]
    return w
    
if __name__ == "__main__":
    #generate data
    x=np.linspace(-1,1,32)
    y=x**2 + np.random.normal(0,0.1,32)
    # fit unregularized model with numpy
    ureg = np.polyfit(x,y,24)
    # fit regularized model
    reg_poly = get_reg_poly_model(x,y,24, alpha=.1)
    fig,ax=plt.subplots(1, figsize=(4,3))
    # build plot
    ax.plot(np.linspace(-1,1,1024), np.polyval(ureg, np.linspace(-1,1,1024)),
    label='unregularized')
    ax.plot(np.linspace(-1,1,1024), np.polyval(reg_poly, np.linspace(-1,1,1024)),
    label='regularized', color='C1')
    # [...] fill in the rest
    ax.scatter(x,y, color='k', s=10)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.set_ylim(-0.5,1.5)
    plt.tight_layout()
    plt.show()
