import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# Box Mueller Method- Returns X and Y ~ N(0,1)
def box_mueller(N,plot=False):

    #Generate N random numbers from a uniform distribution
    U = np.random.uniform(0,1,N)
    V = np.random.uniform(0,1,N)

    #Find Θ and R
    Θ = 2*np.pi*U
    R = np.sqrt(-2*np.log(V))

    #Generate X and Y ~ N(0,1)
    X = R*np.cos(Θ)
    Y = R*np.sin(Θ)

    if plot:
        plot_normal(X,Y)

    return X,Y

# Change Z ~ N(0,1) to Z ~ N(μ,σ^2)
def normal(X,Y,μ,σ,plot=False):
    X = σ*X + μ
    Y = σ*Y + μ

    if plot:
        plot_normal(X,Y)

    return X,Y

# Plotting function
def plot_normal(X,Y):
    #Plots
    fig = plt.figure()
    gs = GridSpec(2, 2, figure=fig) # 2x2 grid

    #Scatter plot of X and Y
    dot_size = 5
    ax = fig.add_subplot(gs[:, 0])
    ax.scatter(X,Y,s=dot_size)
    ax.set_title("Bivariate Normal Distribution")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

    #Histogram of X
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_title("Histogram of X")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Frequency")
    ax1.hist(X, bins=50, density=True, alpha=0.6, color='g')

    #Histogram of Y
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_title("Histogram of Y")
    ax2.set_xlabel("Y")
    ax2.set_ylabel("Frequency")
    ax2.hist(Y, bins=50, density=True, alpha=0.6, color='r')

    plt.show()

#Bivariate Normal Distribution
def bivariate_normal(μ1,μ2,σ1,σ2,σ12,N,plot=False):
    
    """
    Theory Notes 

    X1 ~ N(0,1), X2 ~ N(0,1)
    X = (X1,X2) ~ N2(μ,Σ) is the bivariate normal distribution

    μ = [μ1,μ2] is the Mean Vector
    Σ = [[σ1**2,σ12],[σ12,σ2**2]] is theCovariance Matrix

    σ12 = Cov(X1,X2) = E[(X1-μ1)(X2-μ2)]

    Let A be a 2x2 matrix, Suppose Y = AX
    Then Y ~ N2(Aμ,AΣA^T) is the bivariate normal distribution

    We want to sample from N2(μ,Σ)

    Z1,Z2 ~ N(0,1)
    Z = [Z1,Z2] ~ N2(0,I) where I is the 2x2 identity matrix, 0 = [0,0]

    X = b + AZ ~ N2(μx,Σx) for some vector b and matrix A
    We need to find b and A such that μx = μ and Σx = Σ

    By linearity of expectation,μx = E[X] = E[b] + E[AZ] = b + AE[Z] = b + A*0 = b
    So, μx = b = μ

    By properties of covariance, Σx = Cov(X) = Cov(b + AZ) = Cov(AZ) = A*Cov(Z)*A^T = A*I*A^T = AA^T
    So, Σx = AA^T = Σ

    We need to find A such that AA^T = Σ
    We can use Cholesky Decomposition to find A
    For any symmetric, positive definite matrix Σ, there exists a unique lower triangular matrix L such that Σ = LL^T

    For Bivariate
    The lower triangular matrix A has the form A= [[t11,0],[t21,t22]]
    We want to find t11,t21,t22 such that AA^T = Σ = [[σ1**2,σ12],[σ12,σ2**2]]
    Multiply A and A^T to get Σ and equate the elements to get t11,t21,t22
    Then A = [[t11,0],[t21,t22]]
    X = μ + AZ ~ N2(μ,Σ) """

    # Code

    # Generate standard normal random variables
    Z1, Z2 = box_mueller(N) # or Z1, Z2 = np.random.normal(0,1,N), np.random.normal(0,1,N)

    # Initialize lists to store the results
    X1, X2 = [], []

    # Initialize the Mean Vector and Covariance Matrix
    μ = np.array([μ1, μ2])  # Mean Vector    
    Σ = np.array([[σ1**2, σ12], [σ12, σ2**2]])  # Covariance Matrix

    # Find the Cholesky Decomposition of Σ
    A = cholesky_decomposition(Σ) #or  A = np.linalg.cholesky(Σ)
    
    # Generate the Bivariate Normal Distribution for each pair of Z1 and Z2

    for z1, z2 in zip(Z1, Z2):
        
        Z = np.array([z1, z2])
        X = μ + np.matmul(A, Z)

        X1.append(X[0])
        X2.append(X[1])

    # Plot the Bivariate Normal Distribution
    plot_normal(X1, X2)

# Wrapper functions which takes use input    
def bivariate_input():
    print("Enter μ1,μ2,σ1,σ2,σ12")
    μ1,μ2,σ1,σ2,σ12 = map(float,input().split())
    bivariate_normal(μ1,μ2,σ1,σ2,σ12,N,plot=True)

# Multivariate function - Verifies the bivariate case for n = 2
def multivariate_normal(N,plot=False):

    print("Enter μ and Σ for the Multivariate Normal Distribution")

    μ = list(map(float,input().split()))
    Σ = [list(map(float,input().split())) for i in range(len(μ))]
    A = cholesky_decomposition(Σ) #or  A = np.linalg.cholesky(Σ)
    
    n = len(μ) 
    Z = [box_mueller(N) for z in range(n//2)] # because box_mueller returns 2 random numbers
    # Z = [np.random.normal(0,1,N) for z in range(n)] # or this

    # Reshape to make N (No of samples) columns and n (No of variables Z1,Z2,Z3 ....) rows
    Z = np.array(Z).reshape(n, N)

    # Initialize lists to store the results
    X = np.zeros((n, N))


    # Iterate over the generated random variables
    for i in range(N):
        Z_vec = Z[:, i] # Z vector for the ith sample
        μ_vec = np.array(μ)
        X[:, i] = μ_vec + np.matmul(A, Z_vec) # X = μ + AZ
    
    if plot and n == 2:
        plot_normal(X[0], X[1])
    elif plot:
        print("Plotting is only supported for 2 dimensions :( ")

def cholesky_decomposition(Σ):

    #Cholesky Decomposition Algorithm for any symmetric, positive definite matrix Σ with dimension dim
    #Used Eq 33.8 a,b
    dim = len(Σ)
    A = np.zeros((dim,dim))
    for i in range(dim):
        for j in range(i+1):
            sum = 0
            for k in range(j):
                sum += A[i][k]*A[j][k]
            if i==j:
                A[i][j] = np.sqrt(Σ[i][i]-sum)
            else:
                A[i][j] = (Σ[i][j]-sum)/A[j][j]
    
    return A        

N = 1000 #Number of random numbers to generate

#bivariate_normal(1,1,1,1,0.7,N,plot=True)
bivariate_normal(1,1,1,1,0,N,plot=True)
#bivariate_input()
#multivariate_normal(N,plot=True)