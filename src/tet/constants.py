from tet.data_process import writeData
import os

class Constants:
    def __init__(self):
        self.xA = 0.
        self.xD = 0.
        self.max_N = 0.
        self.max_t = 0
        self.omegaA = 0.
        self.omegaD = 0.
        self.constants = {'xA': self.xA, 'xD': self.xD, 'max_N': self.max_N,\
            'max_t': self.max_t, 'omegaA': self.omegaA, 'omegaD': self.omegaD}
    
    def __call__(self):
        return self.constants
    
    def setConstant(self, key, value):
        self.constants[key] = value
        
    def getConstant(self, key):
        return self.constants[key]
    
    def dumpConstans(self):
        data_list = list(self.constants.items())
        writeData(data=data_list, destination=os.getcwd(), name_of_file='system_constants')