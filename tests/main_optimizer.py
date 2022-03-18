from tet.Solver import solver
import tensorflow as tf
import tet.constants as constants

if __name__=="__main__":
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

    # Make an initial search of the parameter space
    min_a, min_d, loss = solver(a_lims=[-3, 3],
                                d_lims=[-3, 3],
                                grid_size=2, 
                                case=0, 
                                iterations=500, 
                                learning_rate=0.1, 
                                create_plot=True)
    
    a_min, a_max = min_a-1, min_a+1
    d_min, d_max = min_d-1, min_d+1
    xmin, xmax, loss = solver(a_lims=[a_min, a_max], 
                              d_lims=[d_min, d_max], 
                              grid_size=4, 
                              case=1, 
                              iterations=1000,
                              learning_rate=0.01, 
                              create_plot=True)