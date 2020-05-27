import numpy as np
from functools import partial
from collections import deque
from IPython.core.debugger import set_trace
from scipy.linalg import block_diag
from scipy.optimize import minimize
from tqdm import tqdm
import time
from rk4 import rkStep, rksol
from scipy.linalg import block_diag
from numpy.linalg import inv

######## Graph stuff
def ALine(n):
    tuple_rem = ((0, 0), (-1, -1))
    A = np.roll(np.eye(n), 1) + np.roll(np.eye(n), -1)
    for pos in tuple_rem:
        A[pos] = 0
    #A += np.eye(n)
    return A

def DegreeLine(n):
    degree = 2*np.eye(n)
    degree[0, 0] = 1
    degree[-1, -1] = 1
    return degree

def LaplLine(n):
    return DegreeLine(n) - ALine(n)

def ACircle(n):
    A = ALine(n)
    A[0, -1] = 1
    A[-1, 0] = 1
    return A

def DegreeCircle(n):
    return 2*np.eye(n)

def LaplCircle(n):
    return DegreeCircle(n) - ACircle(n)


##### Accelerated Gradient Systems
def rk4Continuous(F, tspan, y0, dt):
    t = 0
    timeDomain = deque([t])
    y = y0
    ysol = deque([y0])
    pbar = tqdm(total = (tspan[1]-tspan[0])/dt + 1)
    while t < tspan[1]:
        y = rkStep(F, t, y, dt)
        t += dt
        timeDomain.append(t)
        ysol.append(y)
        pbar.update(1)
    return rksol(np.array(ysol), np.array(timeDomain))

def rk4HybridSyncClocksAcc(F, G, tspan, y0, dt, checkFlow, checkJump, tqdm = tqdm):
    flow = True 
    t = 0
    j = 0
    y = y0
    timeDomain = deque([[t,j]])
    ysol = deque([y0])
    pbar = tqdm(total = (tspan[1]-tspan[0])/dt + 1, desc= 'Sim')
    while t < tspan[1]:
        while flow and t < tspan[1]:
            y = rkStep(F, t, y, dt)
            t += dt
            timeDomain.append([t,j])
            ysol.append(y)
            flow, triggerIndices = checkFlow(y)
            pbar.update(1)
        while not flow and t < tspan[1]:
            y = G(y, triggerIndices)
            j += 1
            timeDomain.append([t,j])
            ysol.append(y)
            jump, triggerIndices = checkJump(y)
            flow = not jump
    return rksol(np.array(ysol), np.array(timeDomain))


def FDistGradientCortes(y, t, L, n, p, gradF):
    x = y[:n*p]
    z = y[n*p:]
    xdot = -L@x - L@z - gradF(x)
    zdot = L@x
    return np.vstack([xdot, zdot])

def FAcc(x, t, L, zstar, n, p, gamma = 1/4, disturbance = None):
    #tic = time.perf_counter()
    x1 = x[:n*p]
    x2 = x[n*p: 2*n*p]
    tau = x[2*n*p:2*n*p + n]
    Dtau = np.diag(np.kron(tau, np.ones((p,1))).flatten())
    DtauInv = np.diag(np.kron(1/tau, np.ones((p, 1))).flatten())
    x1dot = 2*DtauInv@(x2-x1)
    ## This is not actually the gradient as the
    ## multiplication between L and D(tau) not 
    ## necessarily commutes.
    if disturbance is None:
        gradient = L@Dtau@zstar(L@x1)
    else:
        gradient = L@Dtau@zstar(L@(x1 + disturbance(t))) 
    x2dot = -2*gamma*gradient
    taudot = 0.5*np.ones((n, 1))
    #toc = time.perf_counter()    
    #print('Time FAccCent:', toc - tic)
    return np.vstack([x1dot, x2dot, taudot]).reshape(-1, 1)

def FAccNonDist(x, t, L, zstar, n, p, gamma = 1/4, disturbance = None):
    #tic = time.perf_counter()
    x1 = x[:n*p]
    x2 = x[n*p: 2*n*p]
    tau = x[2*n*p:2*n*p + n]
    Dtau = np.diag(np.kron(tau, np.ones((p,1))).flatten())
    DtauInv = np.diag(np.kron(1/tau, np.ones((p, 1))).flatten())
    x1dot = 2*DtauInv@(x2-x1)
    ## This is not actually the gradient as the
    ## multiplication between L and D(tau) not 
    ## necessarily commutes.
    if disturbance is not None:
        gradient = L@Dtau@zstar(L@x1)
    else:
        gradient = L@Dtau@zstar(L@x1) + disturbance(t)
    x2dot = -2*gamma*gradient
    taudot = 0.5*np.ones((n, 1))
    #toc = time.perf_counter()    
    #print('Time FAccCent:', toc - tic)
    return np.vstack([x1dot, x2dot, taudot]).reshape(-1, 1)

# This has the laplacian after the diagonal matrix of times
def FAccAlt(x, t, L, zstar, n, p, gamma = 1/4):
    #tic = time.perf_counter()
    x1 = x[:n*p]
    x2 = x[n*p: 2*n*p]
    tau = x[2*n*p:2*n*p + n]
    Dtau = np.diag(np.kron(tau, np.ones((p,1))).flatten())
    DtauInv = np.diag(np.kron(1/tau, np.ones((p, 1))).flatten())
    x1dot = 2*DtauInv@(x2-x1)
    x2dot = -2*gamma*Dtau@L@zstar(L@x1)
    taudot = 0.5*np.ones((n, 1))
    #toc = time.perf_counter()    
    #print('Time FAccCent:', toc - tic)
    return np.vstack([x1dot, x2dot, taudot]).reshape(-1, 1)

## Strong 0 
#### x1+ = x1, x2+ = x2
## Strong 1
#### x1+ = x1, x2+ = x1
## Strong 2
#### x1+ = x2, x2+ = x2

def GAcc(x, iTrigger, tmin, tmax, rvec, A, n, p, strong = 0):
    #tic = time.perf_counter()
    x1 = x[:n*p]
    x2 = x[n*p: 2*n*p]
    tau = x[2*n*p:2*n*p + n]
    
    ## xplus
    
    if strong == 0:
        x1Plus = np.copy(x1)
        x2Plus = np.copy(x2)
    elif strong == 1:
        x1Plus = np.copy(x1)
        x2Plus = x1Plus
    elif strong == 2:
        x2Plus = np.copy(x2)
        x1Plus = x2Plus
    else:
        raise ValueError(f'Strong = {strong} not supported.')

    ## Tau plus
    ## Potential tau plus
    pXPlus = (tmax-tmin)*(tau > rvec).astype(int) + tmin 
    equalVec = (tau == rvec)
    pXPlus[equalVec] = (tmax-tmin)*(np.random.rand(int(np.sum(equalVec))) > 0.5).astype(int)  + tmin 
    tauPlus = np.copy(tau)
    indexUpdate = iTrigger.pop()
    ## TODO: implement adjacency list instead of adjacency matrix
    ## for more efficient iterations
    for i, isNeighbor in enumerate(A[indexUpdate, :]):
        ## Check if is connected
        if i != indexUpdate and bool(isNeighbor):
            tauPlus[i] = pXPlus[i]               
    tauPlus[indexUpdate] = tmin
    #toc = time.perf_counter() ``   
    #print('Time GAccCent:', toc - tic)
    return np.vstack([x1Plus, x2Plus, tauPlus]).reshape(-1, 1)

def GAccNoSync(x, iTrigger, tmin, tmax, rvec, A, n, p, strongly = True):
    #tic = time.perf_counter()
    x1 = x[:n*p]
    x2 = x[n*p: 2*n*p]
    tau = x[2*n*p:2*n*p + n]
    
    ## xplus
    x1Plus = np.copy(x1)
    x2Plus = np.copy(x2)

    indexUpdate = iTrigger.pop()
    tauPlus = np.copy(tau)
    tauPlus[indexUpdate] = tmin
    #toc = time.perf_counter() ``   
    #print('Time GAccCent:', toc - tic)
    return np.vstack([x1Plus, x2Plus, tauPlus]).reshape(-1, 1)    


def checkFlowSyncAcc(x, tmin, tmax, n, p):
    tau = x[2*n*p:2*n*p + n]
    checkVec = np.logical_and(tau < tmax, tau >= tmin)
    return np.all(checkVec), (np.where(checkVec == False)[0]).tolist()

def checkJumpSyncAcc(x, tmin, tmax, n, p):
    tau = x[2*n*p:2*n*p + n]
    checkVec = (tau >= tmax)
    return np.any(checkVec), (np.where(checkVec == True)[0]).tolist()

## Todo change the initial condition
def zstar(x, F):
    #tic = time.perf_counter()
    fOpt = lambda z: F(z) - z@x

    res = minimize(fOpt, np.zeros(x.shape))
    #toc = time.perf_counter()    
    #print('Time zstar:', toc - tic)
    return res.x.reshape(-1, 1)


# ### Primal cost functions
# H is the vstack of the Hi matrices with size lxp
# b is the vstack of the bi vectors with size l
# l is the number of data points per node
# n number of nodes
class linearRegressionProblem(object):
    def __init__(self, n, l, p, L, zpStar = None, H = None, b = None):
        # Generate the data
        if zpStar is None and H is None and b is None:
            self.zPrimalStar =  30*np.random.randn(p).reshape(-1, 1)  + 15## Predetermined optimal solution
            self.H = 3*np.random.randn(l*n, p) + 1.5
            self.b = self.H@self.zPrimalStar + np.sqrt(0.1)*np.random.randn(n*l, 1)
        else:
            print('Preloaded')
            self.zPrimalStar = zpStar
            self.H = H
            self.b = b
        self.Hi = [self.H[i*l:(i+1)*l,:] for i in range(n)]
        self.Hdiag = block_diag(*self.Hi)
        self.HiTdiag = block_diag(*[h.T for h in self.Hi])
        self.HiTHi = block_diag(*[h.T@h for h in self.Hi])
        self.HiTHidiagInv = inv(self.HiTHi)
        self.L = L
        self.p = p
        self.n = n
        self.l = l
    
    def costFunction(self, z):
        z = z.reshape(-1, 1)
        return (1/(2*self.n*self.l))*((self.b-self.Hdiag@z).T)@(self.b-self.Hdiag@z)


    def zstar(self, z):
        return self.HiTHidiagInv@((self.n*self.l*self.L@z).reshape(-1, 1) + self.HiTdiag@self.b)
    
    def grad(self, z):
        return  (1/self.n*self.l)*(self.HiTHi@z - self.HiTdiag@self.b)

# H is the vstack of the Hi matrices with size lxp
# b is the vstack of the bi vectors with size l
# l is the number of data points per node
# n number of nodes
""" class linearRegressionProblem4Cost(object):
    def __init__(self, n, l, p, L, zpStar = None, H = None, b = None):
        # Generate the data
        if zpStar is None and H is None and b is None:
            self.zPrimalStar =  30*np.random.randn(p).reshape(-1, 1)  + 15## Predetermined optimal solution
            self.H = 3*np.random.randn(l*n, p) + 1.5
            self.b = self.H@self.zPrimalStar + np.sqrt(0.1)*np.random.randn(n*l, 1)
        else:
            print('Preloaded')
            self.zPrimalStar = zpStar
            self.H = H
            self.b = b
        self.Hi = [self.H[i*l:(i+1)*l,:] for i in range(n)]
        self.Hdiag = block_diag(*self.Hi)
        self.HiTdiag = block_diag(*[h.T for h in self.Hi])
        self.HiTHi = block_diag(*[h.T@h for h in self.Hi])
        self.HiTHidiagInv = inv(self.HiTHi)
        self.L = L
        self.p = p
        self.n = n
        self.l = l
    
    def costFunction(self, z):
        z = z.reshape(-1, 1)
        return (0.01/(4*self.n*self.l))*np.sum(np.power((self.b-self.Hdiag@z), 4))

    def zstar(self, z):
        cost2 = (0.01/(self.n*self.l))*((self.b-self.Hdiag@z).T)@(self.b-self.Hdiag@z)
        return self.HiTHidiagInv@((1/1)*(self.n*self.l*self.L@z).reshape(-1, 1) + self.HiTdiag@self.b)
    
    def grad(self, z):
        grad2 = 2*(self.HiTHi@z - self.HiTdiag@self.b)
        cost2 =(0.01/(2*self.n*self.l))*((self.b-self.Hdiag@z).T)@(self.b-self.Hdiag@z)
        return (0.01/(self.n*self.l))*cost2*grad2 """