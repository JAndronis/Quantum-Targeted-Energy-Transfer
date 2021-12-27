import pandas as pd
import numpy as np
import os
import shutil

def writeData(data, destination, name_of_file, zip_files=False):
    df = pd.DataFrame(data = data)
    _destination = os.path.join(destination, name_of_file)
    np.savetxt(_destination, df)
    
    if zip_files:
        shutil.make_archive(base_name=destination, format='zip')