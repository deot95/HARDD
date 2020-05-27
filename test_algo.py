import distributedAcceleratedGradient as algo
#import cupy as cp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
import os
from tqdm import tqdm

def my_rc_params(magnifier=1):
    plt.style.use(["seaborn-deep"])
    plt.rcParams['axes.labelsize'] = 12*magnifier
    plt.rcParams['xtick.labelsize'] = 12*magnifier
    plt.rcParams['ytick.labelsize'] = 12*magnifier
    plt.rcParams['legend.fontsize'] = 12*magnifier
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['figure.figsize'] = 12, 6
    plt.rc('text', usetex=True)    
my_rc_params(3)     

def get_measures(y, problemInstance, bigL):
    primalDistCostReg = problemInstance.costFunction
    n, p = problemInstance.n, problemInstance.p
    zHist = y[:,:n*p,:]     
    Fstar = primalDistCostReg(np.kron(np.ones((n, 1)), problemInstance.zPrimalStar))
    costDifference = np.array([primalDistCostReg(z) - Fstar for z in zHist])
    consensusDistance = np.array([np.linalg.norm(bigL@z) for z in zHist])
    return costDifference, consensusDistance

def get_primal_variables(y, problemInstance, bigL):
    n, p = problemInstance.n, problemInstance.p
    x = y[:,:n*p,:]    
    primalDistCostReg = problemInstance.costFunction
    zFromDual = problemInstance.zstar
    zHist = np.array([zFromDual(xi) for xi in tqdm(x, desc= 'Primal Vars')])
    zHist = zHist.reshape(*zHist.shape[:-1])
    Fstar = primalDistCostReg(np.kron(np.ones((n, 1)), problemInstance.zPrimalStar))
    costDifference = np.array([primalDistCostReg(z) - Fstar for z in zHist])
    consensusDistance = np.array([np.linalg.norm(bigL@z) for z in zHist])
    return costDifference, consensusDistance, zHist

### Algo test
tspan = [0, 6.5e2]
tmin, tmax = 0.1, 35
dt = 1e-4
nAgents = 5
l = 10 # l is the number of data points per node
p = 8 # Dimension of the primal decision variable
A = algo.ACircle(nAgents)
L = algo.LaplCircle(nAgents)
bigL = np.kron(L, np.eye(p))
regProb = algo.linearRegressionProblem(nAgents, l, p, bigL)
gamma = 1/4

### Flow map for regular algorithm
FRegression = partial(algo.FAcc, L = bigL, zstar = regProb.zstar,
                         n = nAgents, p = p, gamma = gamma)

### Flow distributed non-accelerated Cortes
FdistNonAcc = partial(algo.FDistGradientCortes, L = bigL,
                         n = nAgents, p = p, gradF = regProb.grad)
                                     

""" ### Flow map for algorithm with exchanged L and D in y dynamics
FRegressionAlt = partial(algo.FAccAlt, L = bigL, zstar = regProb.zstar,
                         n = nAgents, p = p) """

### Synching jump map
rvec = np.array([(tmax - tmin)/(nAgents + 1) + tmin]*nAgents).reshape(-1, 1)


GRegression = partial(algo.GAcc, tmin = tmin, tmax = tmax,
                        rvec = rvec , A = A, n = nAgents, p = p)

GRegx2tox1 = partial(algo.GAcc, tmin = tmin, tmax = tmax,
                        rvec = rvec , A = A, n = nAgents, p = p, strong = 1)
GRegx1tox2 = partial(algo.GAcc, tmin = tmin, tmax = tmax,
                        rvec = rvec , A = A, n = nAgents, p = p, strong = 2)                        

### Non Synching jump map
GRegressionNoSync = partial(algo.GAccNoSync, tmin = tmin, tmax = tmax,
                        rvec = rvec , A = A, n = nAgents, p = p)

### Flow and jump checks
checkFlow = partial(algo.checkFlowSyncAcc, tmin = tmin, tmax = tmax, n = nAgents, p = p)
checkJump = partial(algo.checkJumpSyncAcc, tmin = tmin, tmax = tmax, n = nAgents, p = p)                        
                        

################################################
################################################
####### Simulation and saving
################################################3
################################################3

## Initial conds and other params
x1_0 = 50*np.random.randn(nAgents*p, 1)
x2_0 = 50*np.ones((nAgents*p, 1))
x2_0[::2] = -50
#x2_0 = np.copy(x1_0)
#x2_0[::2] = -3
assert np.ones((nAgents*p,1)).T@x2_0 == 0
t0 = tmin*np.ones((nAgents, 1))
x0 = np.vstack([x1_0, x2_0, t0])

results_path = './results1e4/'
power2 = './power2/'
power4 = './power4/'

sync0_path = results_path + 'sync0/'
no_sync0_path = results_path + 'no_sync0/'
x1tox2 = no_sync0_path + '/x1tox2/'
x2tox1 = no_sync0_path + '/x2tox1/'
cortes_path =  results_path  + "/nonAccDist/"
never_sync_path = results_path + '/never_sync/'
never_resets = results_path + '/never_resets/'
power2 = '/power2/'
power4 = '/power4/'
powers = [power2, power4]

paths = [sync0_path, never_resets,
         no_sync0_path, never_sync_path,
         x1tox2, x2tox1, cortes_path]
paths_full_2 = [path + power2 for path in paths]
paths_full_4 = [path + power4 for path in paths]
for path in [*paths_full_2,*paths_full_4]:
    if not os.path.exists(path):
        os.makedirs(path)


##### quad power Cost
print('Sync 0')
x0 = np.vstack([x1_0, x2_0, t0])
sol = algo.rk4HybridSyncClocksAcc(FRegression, GRegx1tox2, tspan,
                              x0, dt, checkFlow, checkJump)
costDifference, consensusDistance, zHist = get_primal_variables(sol.y, regProb, bigL)
np.savez(sync0_path + power2 + 'dual_vars.npz', y = sol.y, t = sol.t,
                                        Hi = regProb.Hi, b = regProb.b, zstar = regProb.zPrimalStar,
                                        n = nAgents, p = p, l = l, bigL = bigL)
np.savez(sync0_path + power2 + 'primal_vars.npz',
                    cost = costDifference, consensus = consensusDistance,
                    zHist = zHist, t = sol.t)


""" print('Grad Cortes')
z1_0 = regProb.zstar(x1_0)
z2_0 = np.zeros(z1_0.shape)#regProb.zstar(x2_0)
z0 = np.vstack([z1_0, z2_0])
sol = algo.rk4Continuous(FdistNonAcc, tspan, z0, dt)
costDifference, consensusDistance = get_measures(sol.y, regProb, bigL)
## Todo get dual variable values.
np.savez(cortes_path + power2 +'dual_vars.npz' , y = sol.y, t = sol.t,
                                        Hi = regProb.Hi, b = regProb.b, zstar = regProb.zPrimalStar,
                                        n = nAgents, p = p, l = l, bigL = L)
np.savez(cortes_path + power2 + 'primal_vars.npz' , cost = costDifference,
                 consensus = consensusDistance, zHist = sol.y, t = sol.t)                               

### x1 to x2
print('\nNot Sync 0: x1 ->x2')
t0 = np.linspace(tmin, tmax/2, nAgents).reshape(-1, 1)
x0 = np.vstack([x1_0, x2_0, t0])
sol = algo.rk4HybridSyncClocksAcc(FRegression, GRegx1tox2, tspan,
                              x0, dt, checkFlow, checkJump)
costDifference, consensusDistance, zHist = get_primal_variables(sol.y, regProb, bigL)
np.savez(x1tox2 + power2 + 'dual_vars.npz', y = sol.y, t = sol.t,
                                        Hi = regProb.Hi, b = regProb.b, zstar = regProb.zPrimalStar,
                                        n = nAgents, p = p, l = l, bigL = bigL)
np.savez(x1tox2 + power2 + 'primal_vars.npz',
                    cost = costDifference, consensus = consensusDistance,
                    zHist = zHist, t = sol.t)
                         


print('\nNever Sync')
x0 = np.vstack([x1_0, x2_0, t0])
sol = algo.rk4HybridSyncClocksAcc(FRegression, GRegressionNoSync, tspan,
                              x0, dt, checkFlow, checkJump)
costDifference, consensusDistance, zHist = get_primal_variables(sol.y, regProb, bigL)
np.savez(never_sync_path + power2 + 'dual_vars.npz', y = sol.y, t = sol.t,
                                        Hi = regProb.Hi, b = regProb.b, zstar = regProb.zPrimalStar,
                                        n = nAgents, p = p, l = l, bigL = bigL)
np.savez(never_sync_path + power2 + 'primal_vars.npz',
                    cost = costDifference, consensus = consensusDistance,
                    zHist = zHist, t = sol.t)


print('\nNever Resets')
x0 = np.vstack([x1_0, x2_0, t0])
tmin, tmax = 0.1, 2*tspan[1]
checkFlowNever = partial(algo.checkFlowSyncAcc, tmin = tmin, tmax = tmax, n = nAgents, p = p)
checkJumpNever = partial(algo.checkJumpSyncAcc, tmin = tmin, tmax = tmax, n = nAgents, p = p)     
Gnever = partial(algo.GAcc, tmin = tmin, tmax = tmax,
                        rvec = rvec , A = A, n = nAgents, p = p, strong = 2)     
sol = algo.rk4HybridSyncClocksAcc(FRegression, Gnever, tspan,
                              x0, dt, checkFlowNever, checkJumpNever)

costDifference, consensusDistance, zHist = get_primal_variables(sol.y, regProb, bigL)
np.savez(never_resets + power2 + 'dual_vars.npz', y = sol.y, t = sol.t,
                                        Hi = regProb.Hi, b = regProb.b, zstar = regProb.zPrimalStar,
                                        n = nAgents, p = p, l = l, bigL = bigL)
np.savez(never_resets + power2 + 'primal_vars.npz',
                    cost = costDifference, consensus = consensusDistance,
                    zHist = zHist, t = sol.t) """

