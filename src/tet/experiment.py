import tensorflow as tf
import nni
from nni.experiment import Experiment

search_space = {
    'lr': {'_type': 'uniform', '_value':[0.4, 0.9]},
    'beta_1': {'_type': 'uniform', '_value':[0.5, 0.9]},
    'beta_2': {'_type': 'uniform', '_value':[0.5, 0.999]},
    'amsgrad': {'_type': 'choice', '_value':[True, False]},
}

experiment = Experiment('local')
experiment.config.trial_command = 'python3 model.py'
experiment.config.trial_code_directory = '.'
experiment.config.search_space = search_space
experiment.config.tuner.name = 'TPE'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.max_trial_number = 8
experiment.config.trial_concurrency = 4
experiment.run(8080)
experiment.stop()
