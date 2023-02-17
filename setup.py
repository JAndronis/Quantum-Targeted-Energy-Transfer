import src.tet.constants as constants

if __name__=="__main__":
    temporary_tensorflow_params = { key : value for key, value in constants.TensorflowParams.items() if key != 'DTYPE'}
    temporary_solver_params = { key : value for key, value in constants.solver_params.items() if key != 'methods'}
    parameter_dict = {
        'constants': constants.system_constants, 
        'tensorflow_params': temporary_tensorflow_params,
        'solver_params': temporary_solver_params, 
        'limits': [[-10, 10] for _ in range(constants.system_constants['sites'])]
    }
    constants.dumpConstants(dictionary=parameter_dict)