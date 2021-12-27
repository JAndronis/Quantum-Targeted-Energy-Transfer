import numpy as np

def mp_bosons_at_donor_analytical(max_t, max_N, eigvecs, eigvals, initial_state):
    '''
    Function that calculates the average number of bosons in the dimer system.
    The calculation is done based on the equation (25) in our report.

    INPUTS:
    max_t == int(), The maximum time of the experiment.
    max_N == int(), The number of bosons on the donor site.
    eigvecs == np.array(),
                shape == (max_N+1, max_N+1),
                The eigenvectors of the hamiltonian of the system as columns in a numpy array.
    eigvals == np.array(),
                shape == max_N+1,
                The eigenvalues of the hamiltonian of the system.
    initial_state == np.array(),
                        shape == max_N+1,
                        An initial state for the system defined as the normalised version of:
                        np.exp(-(max_N-n)**2) where n is the n-th boson at the donor site.

    OUTPUTS:
    avg_N == list(),
                len() == max_t
                The average number of bosons at the donor.
    '''
    coeff_c = np.zeros(max_N+1, dtype=float)
    for i in range(max_N+1): 
        coeff_c[i] = np.vdot(eigvecs[:,i], initial_state)

    coeff_b = eigvecs

    avg_N = []
    _time = range(0, max_t+1)
    avg_min = max_N

    # while True:
    #   query = input("Run with multiprocessing? [y/n] ")
    #   fl_1 = query[0].lower() 
    #   if query == '' or not fl_1 in ['y','n']: 
    #       print('Please answer with yes or no')
    #   else: break

    # if fl_1 == 'y': 
    #   p = mp.Pool()
    #   _mp_avg_N_calc_partial = partial(_mp_avg_N_calc, max_N=max_N, eigvals=eigvals, coeff_c=coeff_c, coeff_b=coeff_b)
    #   avg_N = p.map(_mp_avg_N_calc_partial, _time)
    #   return avg_N

    # if fl_1 == 'n':
    for t in _time:
        avg_N.append(np.real(_mp_avg_N_calc(t, max_N, eigvals, coeff_c, coeff_b)))
        # Remove if 
        if avg_N[t] < avg_min: avg_min = avg_N[t]
        else: break

    return avg_N

def _mp_avg_N_calc(t, max_N, eigvals, coeff_c, coeff_b):
    sum_j = 0
    for j in range(max_N+1):
        sum_i = sum(coeff_c*coeff_b[j,:]*np.exp(-1j*eigvals*t))
        sum_k = sum(coeff_c.conj()*coeff_b[j,:].conj()*np.exp(1j*eigvals*t)*sum_i)
        sum_j += sum_k*j
    return sum_j