from tet import *

if __name__=="__main__":

    # Set globals
    constants.setConstant('max_N', 4)
    constants.setConstant('max_t', 25)
    constants.setConstant('omegaA', 3)
    constants.setConstant('omegaD', -3)
    constants.setConstant('omegaMid', 0)
    constants.setConstant('coupling', 0.1)
    constants.setConstant('xMid', 0)
    constants.setConstant('sites', 2)
    constants.setConstant('resolution', 5)
    CONST = constants.constants
    constants.dumpConstants()

    solver_mp.solver_mp(xa_lims=[-5,5], xd_lims=[-5,5], const=CONST, target_site='x1')
    exit(0)