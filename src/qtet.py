#!/usr/bin/env python3

import pathlib
from os import makedirs
from datetime import datetime
import argparse
import multiprocessing as mp

import tet.constants
from tet.solver_mp import solver_mp
from tet.data_process import createDir

def run():

    parser = argparse.ArgumentParser(description="python3 array_solver.py")
    parser.add_argument('-p', '--data_path', type=pathlib.Path,
                        required=True, help='Path to create data dir.')
    parser.add_argument('-c', '--constants', type=pathlib.Path,
                        required=False, help='Path to constants json file. If not provided then the application will use the default parameters provided by the tet package.')
    parser.add_argument('-n', '--ncpus', default=mp.cpu_count(), type=int, required=False,
                        help='Number of cpus to use in the process pool. Default option is to use all available cpus.')
    parser.add_argument('-m', '--method', default='bins', choices=tet.constants.solver_params['methods'], type=str,
                        required=False, help='Method of optimization to use. Default option is "bins".')

    cmd_args = parser.parse_args()
    data_path = cmd_args.data_path.joinpath(
        f'data_{datetime.now().strftime("%Y_%h_%d_%T")}')

    try:
        if pathlib.Path.exists(cmd_args.data_path):
            createDir(destination=data_path, replace_query=False)
        else:
            print(
                f"Input data path {cmd_args.data_path} does not exist! Creating it."
            )
            raise OSError()
    except OSError:
        makedirs(data_path)

    temporary_tensorflow_params_dict = {
        key: value for key, value in tet.constants.TensorflowParams.items() if key != 'DTYPE'
    }
    temporary_solver_params_dict = {
        key: value for key, value in tet.constants.solver_params.items() if key != 'methods'
    }
    temporary_solver_params_dict['method'] = cmd_args.method

    if cmd_args.constants != None:
        if not pathlib.Path.exists(cmd_args.constants):
            raise OSError('Provided path does not exist.')
        else:
            parameter_dict = tet.constants.loadConstants(cmd_args.constants)

            # check if parameter dicts are correct
            if parameter_dict['constants'].keys() != tet.constants.system_constants.keys() or parameter_dict['tensorflow_params'].keys() != temporary_tensorflow_params_dict.keys():
                raise KeyError(
                    'Parameter json file has altered keys, please run "make" to reconfigure it.'
                )
            else:
                tet.constants.system_constants = {
                    **tet.constants.system_constants, **parameter_dict['constants']
                }
                tet.constants.TensorflowParams = {
                    **tet.constants.TensorflowParams, **parameter_dict['tensorflow_params']
                }
                tet.constants.solver_params = {
                    **tet.constants.system_constants, **parameter_dict['solver_params']
                }
                lims = parameter_dict['limits']
    else:
        print('\n* No json file was provided, using default parameters. *\n')
        lims = [[-10, 10]
                for _ in range(tet.constants.system_constants['sites'])]

    # Set sites to optimize
    keys = [f'x{k}lims' for k in tet.constants.TensorflowParams['train_sites']]
    trainable_vars_lims = dict(zip(keys, lims))

    result = solver_mp(
        const=tet.constants.system_constants,
        trainable_vars_limits=trainable_vars_lims,
        lr=tet.constants.TensorflowParams['lr'],
        beta_1=tet.constants.TensorflowParams['beta_1'],
        amsgrad=tet.constants.TensorflowParams['amsgrad'],
        target_site=tet.constants.solver_params['target'],
        data_path=data_path,
        method=cmd_args.method,
        write_data=True,
        cpu_count=cmd_args.ncpus
    )

    final_parameters = {
        'constants': result,
        'tensorflow_params': temporary_tensorflow_params_dict,
        'solver_params': temporary_solver_params_dict
    }

    tet.constants.dumpConstants(
        dictionary=final_parameters, path=data_path, name='parameters'
    )

if __name__=="__main__":
    run()