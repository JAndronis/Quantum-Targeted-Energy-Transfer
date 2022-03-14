from tet.data_process import writeData
import os

class Constants:
    def __init__(self):
        self.xA = 0.
        self.xD = 0.
        self.max_N = 4
        self.max_t = 25
        self.omegaA = 3.
        self.omegaD = -3.
        self.coupling = 0.1
        self.plot_res = 100
        self.constants = {'xA': self.xA, 'xD': self.xD, 'max_N': self.max_N,\
            'max_t': self.max_t, 'omegaA': self.omegaA, 'omegaD': self.omegaD,\
            'coupling': self.coupling, 'resolution': self.plot_res}
    
    def __call__(self):
        return self.constants
    
    def setConstant(self, key, value):
        self.constants[key] = value
        
    def getConstant(self, key):
        return self.constants[key]
    
    def dumpConstants(self):
        data_list = list(self.constants.items())
        writeData(data=data_list, destination=os.getcwd(), name_of_file='system_constants')