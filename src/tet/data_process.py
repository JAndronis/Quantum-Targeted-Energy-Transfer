import numpy as np
import os
import shutil
import sys
from keras.models import model_from_json

def writeData(data, destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)
    np.savetxt(_destination, data)

def read_1D_data(destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)
    data = []
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        data.append(float(lines[0]))
    return data

def read_2D_data(destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)
    X,Y = [],[]
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        X.append(float(lines[0]))
        Y.append(float(lines[1]))
    return X,Y

def compress(zip_files, destination):
    if not zip_files: return None
    if zip_files:
        shutil.make_archive(base_name=f"{destination}-zipped", format='zip')
        shutil.rmtree(path=destination)

def createDir(destination, replace=True):
    if replace:    
        try:
            os.mkdir(destination)
        except OSError as error:
            print(error)
            while True:
                query = input("Directory exists, replace it? [y/n] ")
                fl_1 = query[0].lower() 
                if query == '' or not fl_1 in ['y','n']: 
                    print('Please answer with yes or no')
                else:
                    shutil.rmtree(destination)
                    os.makedirs(destination)
                    break
            if fl_1 == 'n': sys.exit(0)
    else:
        os.makedirs(destination, exist_ok=True)


def SaveWeights(ModelToSave,jsonFileToSave,h5FileToSave):
    #Save weights
    model_json = ModelToSave.to_json()
    with open(jsonFileToSave, "w") as json_file:
      json_file.write(model_json)
    # serialize weights to HDF5
    ModelToSave.save_weights(h5FileToSave)
    print('Saved')
    
def LoadModel(jsonFileToRead,h5FileToRead):
    #load json and create model
    json_file = open(jsonFileToRead, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(h5FileToRead)
    print("Loaded model from disk")

    return loaded_model
