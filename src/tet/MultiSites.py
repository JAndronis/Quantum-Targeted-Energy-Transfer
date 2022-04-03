from itertools import product
# import constraint
import numpy as np
from math import factorial
# from scipy.linalg import block_diag
# from Hamiltonian import Hamiltonian
from matplotlib import pyplot as plt

#Solve the problem to find all the combinations
class DeriveCombinations():

    def __init__(self,N,f):
        self.N = N
        self.f = f
        self.range = [np.arange(N+1) for _ in range(f)]
        self.dim = factorial(N+f-1)//(factorial(f-1)*factorial(N))

    # def SetConstraint(self, *args):
    #     if sum(args) == self.N: return True

    def derive(self):
        # #Set the problem
        # problem = constraint.Problem()
        # #Add the variables. In our case one variable per site
        # variables = ["x{}".format(i) for i in range(f)]

        # for variable in variables: problem.addVariable(variable,np.arange(self.N+1))
        # #Add the constraint
        # problem.addConstraint(self.SetConstraint, variables)
        # #Find the solution of the problem
        # solutions = problem.getSolutions()
        
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


class CreateHamiltonian:

    def __init__(self, maxN, coupling_lambda, Sites, omegas, chis):
        self.maxN = maxN
        self.Sites = Sites
        self.coupling_lambda = coupling_lambda

        self.Nstates = int( factorial(self.maxN+self.Sites-1)/( factorial(self.maxN)*factorial(self.Sites-1) ) )
        self.CombinationsBosons = DeriveCombinations(N=self.maxN,f=self.Sites).derive()
        self.StatesDictionary = dict(zip(np.arange(self.Nstates,dtype= int),self.CombinationsBosons))
        
        # self.omegas = np.random.randint(low=-5, high=5, size=self.Sites)
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
            sum_j += sum_k*self.StatesDict[j][self.target_state]
        return sum_j.real

    def Execute(self):
        Data = []
        self.SetCoeffs()
        #print(self.bcoeffs)
        for t in range(tmax+1):
            x = self._computeAverageCalculation(t)
            Data.append(x)

        return Data

if __name__=="__main__":
    #----------------------------------------------------------------------------------
    #np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)
    #H_final = block_diag(*Hdiagonals)
    #CombinationsBosonsList = [item for sublist in CombinationsBosonsList for item in sublist]
    #Parameters of the problem
    max_N = 2
    f = 3
    coupling = 0.1
    tmax = 300
    omegas = [-3, 1, 3]
    chis = [1.5, 0, -1.5]
    chis_md = np.zeros((3,50))
    chis_md[0,:] = np.linspace(-10,10,50)
    chis_md[1,:] = np.linspace(-10,10,50)
    dim = factorial(max_N+f-1)//(factorial(f-1)*factorial(max_N))
    
    H = np.zeros((len(list(product(chis_md[0,:], chis_md[1,:]))), dim, dim))
    # eigenvalues = np.zeros((len(list(product(chis_md[0,:], chis_md[1,:]))), dim))
    # eigenvectors = np.zeros((len(list(product(chis_md[0,:], chis_md[1,:]))), dim, dim))

    # avg_ND_analytical = np.zeros((eigenvalues.shape[0], tmax+1))
    min_n = np.zeros(len(chis_md[0,:])*len(chis_md[1,:]))
    counter = 0
    param_id = []
    for combination in product(chis_md[0,:], chis_md[1,:]):
        x,y = combination
        temp_H, temp_States = CreateHamiltonian(maxN=max_N,
                                                coupling_lambda=coupling,
                                                Sites=f,
                                                omegas=omegas,
                                                chis=[x, 0, y]).Execute()
        H[counter] = temp_H
        # eigenvalues[counter], eigenvectors[counter] = np.linalg.eigh(H[counter])
        # min_n[counter] = min(Loss(H=temp_H, States=temp_States, maxN=max_N, target_state=f'x{0}').Execute())
        counter += 1
        param_id.append(temp_States)
        print(f'H_{counter} of {len(list(product(chis_md[0,:], chis_md[1,:])))}', end='\r')
    print("\n")
    
    counter = 0
    for i in range(len(chis_md[0,:])):
        for j in range(len(chis_md[1,:])):
            min_n[i*50+j] = min(Loss(H=H[counter], States=param_id[counter], maxN=max_N, target_state=f'x{0}').Execute())
            counter += 1
            print(f'done with combination {counter} of {len(list(product(chis_md[0,:], chis_md[1,:])))}', end='\r')
    print()
    #Create the Hamiltonian of the problem
    # H[], States = CreateHamiltonian(maxN=max_N, 
    #                               coupling_lambda=coupling, 
    #                               Sites=f,
    #                               omegas=omegas,
    #                               chis=chis).Execute()
    # eigenvalues, eigenvectors = np.linalg.eigh(H)
    

    # problemHamiltonian = Hamiltonian(chiA=-0.5, chiD=0.5, coupling_lambda=coupling_lambda, 
    #                                 omegaA = 3, omegaD = -3,max_N=maxN).createHamiltonian()

    # eigenvalues1, eigenvectors1 = np.linalg.eigh(problemHamiltonian)

    # proof that the hamiltonian has the correct shape
    # plot the non-zero elements
    # temp = np.zeros(H.shape)
    # for i in range(len(H)):
    #     for j in range(len(H)):
    #         if H[i,j]!=0:
    #             temp[i,j] = 10
    # plt.imshow(temp, cmap='gray_r')
    # plt.title(f'N={maxN}, f={f}')
    # plt.show()

    # for i in range(len(eigenvalues1)):
    #     print(eigenvalues[i],eigenvalues1[i])

    # for i in range(eigenvectors1.shape[0]):
    #     for j in range(eigenvectors1.shape[0]):
    #         print(eigenvectors1[i][j],eigenvectors[i][j])
    #     print('--------------------------------')
    
    # Data = Loss(H=problemHamiltonian,TotalCombinationsList=CombinationsBosonsList,maxN=maxN).Execute()
    
    XA, XD = np.meshgrid(chis_md[0,:], chis_md[1,:])
    figure, ax = plt.subplots(figsize=(7,7))
    plot = ax.contourf(XD, XA, min_n.reshape(int(np.sqrt(len(min_n))), int(np.sqrt(len(min_n)))), levels=20, cmap='rainbow')
    ax.set_xlabel(r"$\chi_{D}$", fontsize=20)
    ax.set_ylabel(r"$\chi_{A}$", fontsize=20)
    figure.colorbar(plot)
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


