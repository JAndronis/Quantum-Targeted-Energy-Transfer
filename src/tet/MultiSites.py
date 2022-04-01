import constraint
import numpy as np
from math import factorial

from sympy import HadamardPower
from Hamiltonian import Hamiltonian

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


class CreateSubHamiltonian:

    def __init__(self,Nbosons,coupling_lambda,Sites):
        self.NBosons = Nbosons
        self.Sites = Sites
        self.coupling_lambda = coupling_lambda

        self.Nstates = int( factorial(self.NBosons+self.Sites-1)/( factorial(self.NBosons)*factorial(self.Sites-1) ) )
        self.CombinationsBosons = DeriveCombinations(N=self.NBosons,f=self.Sites).SetProblem()
        #print(self.CombinationsBosons)
        self.StatesDictionary = dict(zip(np.arange(self.Nstates,dtype= int),self.CombinationsBosons))
        
        self.omegas = [3,2]
        self.chis = [0.5,1]


    def ConstructElement(self,n,m):
        #Find the Hnm element of the Hamiltonian

        #First Term. Contributions due to the kronecker(n,m) elements
        Term1 = 0

        if n==m:
            for k in range(self.Sites):
                Term1 += self.omegas[k]*self.StatesDictionary[m]["x{}".format(k)] +\
                0.5*self.chis[k]*self.StatesDictionary[m]["x{}".format(k)]**2

        #Second Term.Various contributions
        
        Term2a,Term2b = 0,0
        for k in range(self.Sites-1):
            #print(k)
            #Find the number of bosons
            nk = self.StatesDictionary[m]["x{}".format(k)]
            nkplusone = self.StatesDictionary[m]["x{}".format(k+1)]
            
            #Important check
            if (nkplusone != self.NBosons) and (nk != 0): 
                #Find the new state/vol1
                m1tildastate = self.StatesDictionary[m].copy()
                m1tildastate["x{}".format(k)] = nk-1
                m1tildastate["x{}".format(k+1)] = nkplusone+1

                m1tilda = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m1tildastate)]
            
                if m1tilda == n: Term2a += -self.coupling_lambda*np.sqrt((nkplusone+1)*nk)
            
            if (nkplusone != 0) and (nk != self.NBosons): 
                #Find the new state/vol2
                m2tildastate = self.StatesDictionary[m].copy()
                m2tildastate["x{}".format(k)] = nk+1
                m2tildastate["x{}".format(k+1)] = nkplusone-1

                m2tilda = list(self.StatesDictionary.keys())[list(self.StatesDictionary.values()).index(m2tildastate)]

                if m2tilda == n: Term2b += -self.coupling_lambda*np.sqrt(nkplusone*(nk+1))

            
        Term2 = Term2a + Term2b
            
        return Term1 + Term2


    def Execute(self):
        Hsub = np.zeros(shape=(self.Nstates,self.Nstates),dtype=float )
        for n in range(self.Nstates):
            for m in range(self.Nstates):
                Hsub[n][m] = self.ConstructElement(n=n,m=m)
        #print(Hsub)
        return Hsub


class CreateTotalHamiltonian:

    def __init__(self,maxN,coupling_lambda,Sites):
        self.maxN = maxN
        self.coupling_lambda = coupling_lambda
        self.Sites = Sites



    def Create(self):
        Hdiagonals = []
        Hdiagonals.append([[0]])
        for N in range(1,self.maxN+1):
            #print(N)
            Hsub = CreateSubHamiltonian(Nbosons=N,
                                    coupling_lambda=self.coupling_lambda,
                                    Sites=self.Sites).Execute()
            Hdiagonals.append(Hsub)

        return Hdiagonals


def TotalHamiltonian(*args):
    return block_diag(args)
#----------------------------------------------------------------------------------
from scipy.linalg import block_diag
maxN=3
f=2
coupling_lambda = 0.1

Hdiagonals = CreateTotalHamiltonian(maxN=maxN,coupling_lambda=coupling_lambda,Sites=f).Create()
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
H_final = block_diag(*Hdiagonals)
#H_final = TotalHamiltonian(Hdiagonals)

problemHamiltonian = Hamiltonian(chiA=1, chiD=0.5, coupling_lambda=coupling_lambda, 
                                omegaA=2, omegaD=3, max_N=maxN).createHamiltonian()
eigenvalues1, eigenvectors1 = np.linalg.eigh(problemHamiltonian)


eigenvalues2, eigenvectors2 = np.linalg.eigh(H_final)
