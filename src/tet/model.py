from constants import constants, acceptor
from Optimizer import Optimizer

import nni
import tensorflow as tf
import os

params = {
    'lr': 0.5,
    'beta_1': 0.6,
    'beta_2': 0.699,
    'amsgrad': False,
}

# optimized_params = nni.get_next_parameter()
# params.update(optimized_params)

adam = tf.keras.optimizers.Nadam(
    learning_rate=params['lr'], beta_1=params['beta_1'], 
    beta_2=params['beta_2']
)

n = 4
chi = (constants['omegas'][-1] - constants['omegas'][0])/n
if constants['omegas'][0] < 0: 
    chi_a = -chi
    chi_d = chi
else:
    chi_a = chi
    chi_d = -chi

opt = Optimizer(
    target_site=acceptor,
    DataExist=False, 
    Print=True,
    const=constants,
    opt=adam
)

results = opt(chi_d, 0, chi_a, write_data=False)
# nni.report_final_result(results['loss'][-1])
