from tet.Solver import solver
import tensorflow as tf

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

    done = False
    edge = 1
    iteration = 0
    grid = 2
    # Make an initial search of the parameter space
    min_a, min_d, loss = solver(a_lims=[-3, 3],
                                d_lims=[-3, 3],
                                grid_size=grid, 
                                case=0, 
                                iterations=500, 
                                learning_rate=0.1, 
                                create_plot=True)
    iteration += 1
    if loss<=0.1:
        print('TET!')
        print(min_a, min_d, loss)
        done = True
    else:
        while not done:
            if iteration>3:
                done = True
                break
            edge /= iteration
            if grid<6: grid += 2
            else: continue
            a_min, a_max = min_a-edge, min_a+edge
            d_min, d_max = min_d-edge, min_d+edge
            min_a, min_d, loss = solver(a_lims=[a_min, a_max], 
                                        d_lims=[d_min, d_max], 
                                        grid_size=grid, 
                                        case=iteration, 
                                        iterations=1000,
                                        learning_rate=0.01, 
                                        create_plot=True)
            iteration += 1
            if loss<=0.1:
                print('TET!')
                print(min_a, min_d, loss)
                done = True