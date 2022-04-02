from itertools import combinations
import constraint
import numpy as np
from math import factorial
from scipy.linalg import block_diag
from sympy import HadamardPower
from Hamiltonian import Hamiltonian
import cmath
from matplotlib import pyplot as plt

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


class CreateHamiltonian:

    def __init__(self,maxN,coupling_lambda,Sites):
        self.maxN = maxN
        self.Sites = Sites
        self.coupling_lambda = coupling_lambda

        self.Nstates = int( factorial(self.maxN+self.Sites-1)/( factorial(self.maxN)*factorial(self.Sites-1) ) )
        self.CombinationsBosons = DeriveCombinations(N=self.maxN,f=self.Sites).SetProblem()
        self.StatesDictionary = dict(zip(np.arange(self.Nstates,dtype= int),self.CombinationsBosons))
        
        self.omegas = [-3,3]
        self.chis =[0.5,-0.5]


    #Find the Hnm element of the Hamiltonian
    def ConstructElement(self,n,m):
        #First Term. Contributions due to the kronecker(n,m) elements
        Term1 = 0
        if n==m:
            for k in range(self.Sites):
                Term1 += self.omegas[k]*self.StatesDictionary[m]["x{}".format(k)] +\
                0.5*self.chis[k]*(self.StatesDictionary[m]["x{}".format(k)])**2

        #Second Term.Various contributions
        
        Term2a,Term2b = 0,0
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

    def __init__(self,H,States,maxN):
        self.H = H
        self.States = States
        self.maxN = maxN
        self.Sites = len(self.States[0])
        
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
        #Extra re-assign(;)
        self.InitialState = self.Identity[self.InitialStateIndex]
        #self.InitialState[self.InitialStateIndex] = self.maxN
        self.InitialState =self.InitialState/np.linalg.norm(self.InitialState)
        

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
            sum_j += sum_k*self.StatesDict[j]['x0']
        return sum_j.real

    def Execute(self):
        Data = []
        self.SetCoeffs()
        #print(self.bcoeffs)
        for t in range(tmax+1):
            x = self._computeAverageCalculation(t)
            Data.append(x)

        return Data


      


    
        


#----------------------------------------------------------------------------------
#np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
#H_final = block_diag(*Hdiagonals)
#CombinationsBosonsList = [item for sublist in CombinationsBosonsList for item in sublist]
#Parameters of the problem
maxN=12
f=2
coupling_lambda = 0.001
tmax = 30000

#Create the Hamiltonian of the problem
H,States = CreateHamiltonian(maxN=maxN,coupling_lambda=coupling_lambda,Sites=f).Execute()
eigenvalues, eigenvectors = np.linalg.eigh(H)

problemHamiltonian = Hamiltonian(chiA=-0.5, chiD=0.5, coupling_lambda=coupling_lambda, 
                                omegaA = 3, omegaD = -3,max_N=maxN).createHamiltonian()

eigenvalues1, eigenvectors1 = np.linalg.eigh(problemHamiltonian)

"""
for i in range(len(eigenvalues1)):
    print(eigenvalues[i],eigenvalues1[i])

for i in range(eigenvectors1.shape[0]):
    for j in range(eigenvectors1.shape[0]):
        print(eigenvectors1[i][j],eigenvectors[i][j])
    print('--------------------------------')
"""
#Data = Loss(H=problemHamiltonian,TotalCombinationsList=CombinationsBosonsList,maxN=maxN).Execute()
Data = Loss(H=H,States=States,maxN=maxN).Execute()
plt.plot(np.arange(0,tmax+1),Data)
plt.show()


