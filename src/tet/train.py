import time
import numpy as np

import tensorflow as tf
assert tf.__version__ >= "2.0"
import keras.backend as K

import constants
from loss import Loss

DTYPE = tf.float32
OPT = tf.keras.optimizers.Adam()

@tf.function
def compute_loss(xA, xD, const):
    return Loss(const=const).loss(xA, xD)

def get_grads(xA, xD, const):
    with tf.GradientTape() as t:
            t.watch([xA, xD])
            loss = compute_loss(xA, xD, const)
    grads = t.gradient(loss, [xA, xD])
    del t
    return grads, loss

@tf.function
def apply_grads(xA, xD, const):
    grads, loss = get_grads(xA, xD, const)
    OPT.apply_gradients(zip(grads, [xA, xD]))
    return loss

def train(ChiAInitial, ChiDInitial, const, max_iter=200, lr=0.01):
    # Reset Optimizer
    K.clear_session()
    for var in OPT.variables():
        var.assign(tf.zeros_like(var))
    K.set_value(OPT.learning_rate, lr)
    
    LAMBDA = const['coupling']
    OMEGA_A = const['omegaA']
    OMEGA_D = const['omegaD']
    MAX_N = const['max_N']
    MAX_T = const['max_t']
    
    mylosses = []
    tol = 1e-8
    xA = tf.Variable(initial_value=ChiAInitial, trainable=True, dtype=DTYPE)
    xD = tf.Variable(initial_value=ChiDInitial, trainable=True, dtype=DTYPE)
    xA_best = tf.Variable(initial_value=0, dtype=DTYPE)
    xD_best = tf.Variable(initial_value=0, dtype=DTYPE)
    mylosses.append(3.9)
    best_loss = MAX_N
    counter = 0
    d_data = []
    a_data = []
    a_error_count = 0
    d_error_count = 0

    t0 = time.time()
    for epoch in range(max_iter):
        xA_init = xA.numpy()
        xD_init = xD.numpy()
        loss = apply_grads(xA, xD, const)
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
        "\nOmega_A:", OMEGA_A,
        "| Omega_D:", OMEGA_D,
        "| N:", MAX_N,
        "| Total timesteps:", MAX_T,
        "| Coupling Lambda:",LAMBDA,
        "\n"+40*"-")
    
    return mylosses, a_data, d_data, xA_best.numpy(), xD_best.numpy()