import time
import numpy as np

import tensorflow as tf
assert tf.__version__ >= "2.0"
import keras.backend as K

from tet.constants import Constants
from tet.loss import Loss

# enable memory growth
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

DTYPE = tf.float32
OPT = tf.keras.optimizers.Adam(learning_rate=0.01)

@tf.function
def compute_loss(xA, xD):
    return Loss().loss(xA, xD)

def get_grads(xA, xD):
    with tf.GradientTape() as t:
            t.watch([xA, xD])
            loss = compute_loss(xA, xD)
    grads = t.gradient(loss, [xA, xD])
    del t
    return grads, loss

@tf.function
def apply_grads(xA, xD):
    grads, loss = get_grads(xA, xD)
    OPT.apply_gradients(zip(grads, [xA, xD]))
    return loss

def train(ChiAInitial, ChiDInitial, constants=Constants(), max_iter=200):
    # Reset Optimizer
    K.clear_session()
    for var in OPT.variables():
        var.assign(tf.zeros_like(var))
    
    LAMBDA = tf.constant(constants.coupling, dtype=DTYPE)
    OMEGA_A = tf.constant(constants.omegaA, dtype=DTYPE)
    OMEGA_D = tf.constant(constants.omegaD, dtype=DTYPE)
    MAX_N = tf.constant(constants.max_N, dtype=DTYPE)
    MAX_T = tf.constant(constants.max_t, dtype=tf.int32)
    
    mylosses = []
    tol = 1e-8
    xA = tf.Variable(initial_value=ChiAInitial, trainable=True, dtype=DTYPE)
    xD = tf.Variable(initial_value=ChiDInitial, trainable=True, dtype=DTYPE)
    xA_best = tf.Variable(initial_value=0, dtype=DTYPE)
    xD_best = tf.Variable(initial_value=0, dtype=DTYPE)
    mylosses.append(3.9)
    best_loss = MAX_N.numpy()
    counter = 0
    d_data = []
    a_data = []
    a_error_count = 0
    d_error_count = 0

    t0 = time.time()
    for epoch in range(max_iter):
        xA_init = xA.numpy()
        xD_init = xD.numpy()
        loss = apply_grads(xA, xD)
        if epoch%100 ==0: print(f'Loss:{loss.numpy()}, xA:{xA.numpy()}, xD:{xD.numpy()}, epoch:{epoch}')
        
        errorA = np.abs(xA.numpy() - xA_init)
        errorD = np.abs(xD.numpy() - xD_init)

        mylosses.append(loss.numpy())
        if mylosses[epoch+1] < min(list(mylosses[:epoch+1])):
            xA_best.assign(xA.numpy())
            xD_best.assign(xD.numpy())
            best_loss = mylosses[epoch+1]

        counter += 1
        if counter%10 == 0:
            d_data.append(xD.numpy())
            a_data.append(xA.numpy())

        if np.abs(loss.numpy()) < tol:
            break
        
        if errorA < tol:
            a_error_count += 1
            if a_error_count > 2:
                print('Stopped training because of xA_new-xA_old =', errorA)
                break

        if errorD < tol:
            d_error_count += 1
            if d_error_count > 2:
                print('Stopped training because of xD_new-xD_old =', errorA)
                break
        
    t1 = time.time()
    dt = t1-t0

    print("\nApproximate value of chiA:", xA_best.numpy(), 
        "\nApproximate value of chiD:", xD_best.numpy(),
        "\nLoss - min #bosons on donor:", best_loss,
        "\nOptimizer Iterations:", OPT.iterations.numpy(), 
        "\nTraining Time:", dt,
        "\n"+40*"-",
        "\nParameters:",
        "\nOmega_A:", OMEGA_A.numpy(),
        "| Omega_D:", OMEGA_D.numpy(),
        "| N:", MAX_N.numpy(),
        "| Total timesteps:", MAX_T.numpy(),
        "| Coupling Lambda:",LAMBDA.numpy(),
        "\n"+40*"-")
    
    return mylosses, a_data, d_data, xA_best, xD_best