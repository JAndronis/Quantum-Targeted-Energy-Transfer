import numpy as np
import os
import shutil
import sys
from keras.models import model_from_json
from os.path import exists


def writeData(data, destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)

    if exists(_destination):
        #print('File exists, overwrite')
        with open(_destination, "a") as f:
            np.savetxt(f, data)
    
    else:np.savetxt(_destination, data)


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

def ReadDeque(destination, name_of_file):
    _destination = os.path.join(destination, name_of_file)
    to_return = []
    for line in open(_destination, 'r'):
        lines = [i for i in line.split()]
        to_return.append( (int(float(lines[0])),int(float(lines[1])),float(lines[2]),
                        int(float(lines[3])),int(float(lines[4]))) )

    return to_return 

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


def SaveWeights(ModelToSave,case,destination):
    #Save weights
    _destinationjson = os.path.join(destination, case + '.json')
    model_json = ModelToSave.to_json()
    with open(_destinationjson, "w") as json_file:
      json_file.write(model_json)
    # serialize weights to HDF5
    _destinationh5 =  os.path.join(destination, case + '.h5')
    ModelToSave.save_weights(_destinationh5)
    

def LoadModel(case,destination):
    #load json and create model
    _destinationjson = os.path.join(destination, case + '.json')
    json_file = open(_destinationjson, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    _destinationh5 = os.path.join(destination, case + '.h5')
    loaded_model.compile(loss='mse', optimizer='adam', metrics=['mse'])
    loaded_model.load_weights(_destinationh5)

    

    return loaded_model
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
