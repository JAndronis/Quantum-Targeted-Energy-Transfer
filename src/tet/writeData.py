import pandas as pd
import numpy as np
import os

def writeData(data, destination, name_of_file):
    df = pd.DataFrame(data = data)
    _destination = os.path.join(destination, name_of_file)
    np.savetxt(_destination, df)