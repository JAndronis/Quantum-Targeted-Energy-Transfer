from itertools import product
import numpy as np
from math import factorial
from matplotlib import pyplot as plt
from tet import constants
from tet.saveFig import saveFig

#! Class for solving the problem of finding all the combinations
class DeriveCombinations():

    def __init__(self,N,f):
        self.N = N
        self.f = f
        self.range = [np.arange(self.N+1) for _ in range(self.f)]
        self.dim = factorial(self.N+self.f-1)//(factorial(self.f-1)*factorial(self.N))

    def derive(self):
        values = [i for i in product(*self.range) if sum(i)==self.N]
        keys = [f"x{i}" for i in range(self.f)]
        solution = []
        kv = []
        for i in range(self.dim):
            temp = []
            for j in range(self.f):
                temp.append([keys[j], values[i][j]])
            kv.append(temp)
            solution.append(dict(kv[i])) 

        return solution


#! Class for deriving the Hamiltonian of the problem 
class CreateHamiltonian:

    def __init__(self, maxN, coupling_lambda, Sites, omegas, chis):
        self.maxN = maxN
        self.Sites = Sites
        self.coupling_lambda = coupling_lambda

        self.Nstates = int( factorial(self.maxN+self.Sites-1)/( factorial(self.maxN)*factorial(self.Sites-1) ) )
        self.CombinationsBosons = DeriveCombinations(N=self.maxN,f=self.Sites).derive()
        self.StatesDictionary = dict(zip(np.arange(self.Nstates,dtype= int),self.CombinationsBosons))
        
        self.omegas = omegas
        # random sample in range Unif[a,b), b > a.
        # self.chis = (3-(-3))*np.random.random_sample(size=self.Sites)+(-3)
        self.chis = chis


    #Find the Hnm element of the Hamiltonian
    def ConstructElement(self,n,m):
        #First Term. Contributions due to the kronecker(n,m) elements
        Term1 = 0
        #Second Term. Various contributions
        Term2a, Term2b = 0,0
        
        if n==m:
            for k in range(self.Sites):
                Term1 += self.omegas[k]*self.StatesDictionary[m]["x{}".format(k)] +\
                    0.5*self.chis[k]*(self.StatesDictionary[m]["x{}".format(k)])**2

        for k in range(self.Sites-1):
            #Find the number of bosons
            nk = self.StatesDictionary[m]["x{}".format(k)]
            nkplusone = self.StatesDictionary[m]["x{}".format(k+1)]
            
            #Term 2a/Important check
            if (nkplusone != self.maxN) and (nk != 0): 
                m1TildaState = self.StatesDictionary[m].copy()
                m1TildaState["x{}".format(k)] = nk-1
                m1TildaState["x{}".format(k+1)] = nkplusone+1

                m1TildaIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m1TildaState)]
            
                if m1TildaIndex == n: Term2a += -self.coupling_lambda*np.sqrt((nkplusone+1)*nk)
            
            #Term 2b/Important check
            if (nkplusone != 0) and (nk != self.maxN): 
                #Find the new state/vol2
                m2TildaState = self.StatesDictionary[m].copy()
                m2TildaState["x{}".format(k)] = nk+1
                m2TildaState["x{}".format(k+1)] = nkplusone-1

                m2TildaIndex = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m2TildaState)]

                if m2TildaIndex == n: Term2b += -self.coupling_lambda*np.sqrt(nkplusone*(nk+1))
            
        return Term1 + Term2a + Term2b


    def Execute(self):
        H = np.zeros(shape=(self.Nstates,self.Nstates),dtype=float )
        for n in range(self.Nstates):
            for m in range(self.Nstates):
                H[n][m] = self.ConstructElement(n=n,m=m)
        return H,self.CombinationsBosons 

class Loss:

    def __init__(self, H, States, maxN, target_state='x0'):
        self.H = H
        self.States = States
        self.maxN = maxN
        self.Sites = len(self.States[0])
        self.target_state = target_state
        
        self.dim = len(self.H)
        self.StatesDict = dict(zip(np.arange(self.dim,dtype= int),self.States))
        self.Identity = np.identity(n=self.dim)  
        self.ccoeffs,self.bcoeffs = np.zeros(self.dim),np.zeros(self.dim)
        
    #Derive the initial state
    def DeriveInitialState(self):

        #Assign initially all the bosons at the donor
        self.InitialState = self.States[0]
        self.InitialState = dict.fromkeys(self.InitialState, 0)
    
        self.InitialState['x0'] = self.maxN
        #Find the index
        self.InitialStateIndex = list(self.StatesDict.keys())[list(self.StatesDict.values()).index(self.InitialState)]
        self.InitialState = self.Identity[self.InitialStateIndex]
        
    def SetCoeffs(self):
        self.DeriveInitialState()
        
        self.eigvals, self.eigvecs = np.linalg.eigh(self.H)
        #b coeffs
        self.bcoeffs = self.eigvecs
        #c coeffs
        for i in range(self.dim):self.ccoeffs[i] = np.vdot(self.eigvecs[:,i],self.InitialState)

    def _computeAverageCalculation(self, t):
        sum_j = 0
        for j in range(self.dim):
            sum_i = sum(self.ccoeffs*self.bcoeffs[j,:]*np.exp(-1j*self.eigvals*t))
            sum_k = sum(self.ccoeffs.conj()*self.bcoeffs[j,:].conj()*np.exp(1j*self.eigvals*t)*sum_i)
            sum_j += sum_k*self.StatesDict[j][self.target_state]
        return sum_j.real

    def Execute(self):
        Data = []
        self.SetCoeffs()
        for t in range(tmax+1):
            #print('\r t = {}'.format(t),end="")
            x = self._computeAverageCalculation(t)
            Data.append(x)

        return Data


def CreateHeatmap(max_N,f,coupling,omegas,lims):
    #Create a grid with various values of xA,xD given that the intermediate layer is linear.
    chis_md = np.zeros(shape = (f,grid_size))
    chis_md[0,:] = np.linspace(lims[0],lims[1],grid_size)
    chis_md[1,:] = np.linspace(lims[2],lims[3],grid_size)
    dim = factorial(max_N+f-1)//(factorial(f-1)*factorial(max_N))
    combinations_grid = list(product(chis_md[0,:], chis_md[1,:]))

    H = np.zeros(shape = ( len(combinations_grid), dim, dim) )
    min_n = np.zeros(len(combinations_grid))
    param_id = []

    
    for counter,combination in enumerate(combinations_grid):
        print('Counter = {} out of {}'.format(counter,len(combinations_grid)), end='\r')
        #Combinations of xA,xD
        xD,xA = combination
        #Derive temporal data
        temp_H, temp_States = CreateHamiltonian(maxN=max_N,
                                                coupling_lambda=coupling,
                                                Sites=f,
                                                omegas=omegas,
                                                chis = [xD,0,xA]).Execute()

        min_n[counter] = min(Loss(H=temp_H, States=temp_States, maxN=max_N, target_state=f'x{0}').Execute())
        
        #Store data
        H[counter] = temp_H
        param_id.append(temp_States)
    print()

   
    XA, XD = np.meshgrid(chis_md[0,:], chis_md[1,:])

    figure, ax = plt.subplots(figsize=(5.5,5.5))
    plot = ax.contourf(XD, XA, min_n.reshape(grid_size,grid_size), levels=20, cmap='rainbow')
    ax.set_xlabel(r"$\chi_{D}$", fontsize=20)
    ax.set_ylabel(r"$\chi_{A}$", fontsize=20)
    figure.colorbar(plot)
    if f==3:title = f'N:{max_N},[$\omega_D,\omega_I,\omega_A$]:{omegas},coupling:{coupling}'
    else: title = f'N:{max_N},[$\omega_D,\omega_A$]:{omegas},coupling:{coupling}'
    plt.title(title)
    plt.show()


if __name__=="__main__":
    #Parameters of the problem
    max_N = 2
    f = 3
    coupling = 0.1
    tmax = 200
    omegas = [-3,3,3]
    chis = [1.1287,0,4.3631]
    #Parameters of the grid
    grid_size = 50
    minxDgrid,maxXDgrid = -20,20
    minxAgrid,maxXAgrid = -20,20
    lims = [minxDgrid,maxXDgrid,minxAgrid,maxXAgrid]

    constants.setConstant('max_N', max_N)
    constants.setConstant('max_t', tmax)
    constants.setConstant('omegaA', omegas[0])
    constants.setConstant('omegaD', omegas[-1])
    constants.setConstant('coupling', 0.1)
    constants.setConstant('sites', 3)
    constants.setConstant('resolution', grid_size)
    constants.dumpConstants()

    #Heatmap
    #CreateHeatmap(max_N=max_N,f=f,coupling=coupling,omegas=omegas,lims=lims)
    #Time evolution Donor 2 layers
    evolve_donor = False
    if evolve_donor:
        H,States = CreateHamiltonian(maxN=max_N,coupling_lambda=coupling,Sites=f,omegas=omegas,chis=chis).Execute()
        data = Loss(H=H, States=States, maxN=max_N, target_state=f'x{0}').Execute()
        plt.plot(np.arange(0,tmax+1))
        plt.show()
    #Time evolution:Multisites
    multi_sites_evolve = True
    if multi_sites_evolve:
        H,States = CreateHamiltonian(maxN=max_N,coupling_lambda=coupling,Sites=f,omegas=omegas,chis=chis).Execute()
        Titles = [r"$<N_{D}(t)>$",r"$<N_{I}(t)>$",r"$<N_{A}(t)>$"]
        for i in range(f):
            data_case = Loss(H=H, States=States, maxN=max_N, target_state=f'x{i}').Execute()
            plt.plot(np.arange(0,tmax+1),data_case,label=Titles[i])
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel(r"$<N(t)>$")
        plt.title(f'[$x_D,x_I,x_A$]:{chis},[$\omega_D,\omega_I,\omega_A$]:{omegas},N:{max_N},coupling:{coupling}')
        plt.show()
            
         


    

    # data = []
    # for i in range(f):
    #     _data = Loss(H=H, States=States, maxN=max_N, target_state=f'x{i}').Execute()
    #     data.append(_data)
        
    #     if i==0:
    #         name = 'Donor'
    #     elif i==f-1:
    #         name = 'Acceptor'
    #     elif i==1:
    #         name = f'{i}-st in-between level'
    #     elif i==2:
    #         name = f'{i}-nd in-between level'
    #     elif i==3:
    #         name = f'{i}-rd in-between level'
    #     else:
    #         name = f'{i}-th in-between level'
    #     plt.plot(np.arange(0, tmax+1), _data, label=name)
    # plt.legend()
    # plt.title(f'Time Evolution of n for all levels')
    # plt.xlabel('Timestep')
    # plt.ylabel('n')
    # plt.show()


"""
#? Derive combinations with Constraint Method

#Solve the problem to find all the combinations
class DeriveCombinations():

    def __init__(self,N,f):
        self.N = N
        self.f = f

    def SetConstraint(self,*args):
        if sum(args) == self.N:return True


    def SetProblem(self):
        #Set the problem
        problem = constraint.Problem()
        #Add the variables. In our case one variable per site
        variables = ["x{}".format(i) for i in range(f)]

        for variable in variables: problem.addVariable(variable,np.arange(self.N+1))
        #Add the constraint
        problem.addConstraint(self.SetConstraint, variables)
        #Find the solution if the problem
        solutions = problem.getSolutions()

        return solutions
"""

#? proof that the hamiltonian has the correct shape
# plot the non-zero elements
# temp = np.zeros(H.shape)
# for i in range(len(H)):
#     for j in range(len(H)):
#         if H[i,j]!=0:
#             temp[i,j] = 10
# plt.imshow(temp, cmap='gray_r')
# plt.title(f'N={maxN}, f={f}')
# plt.show()
