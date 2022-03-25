import numpy as np

def conf_r(t, n_pulls):
    """ compute confidence radius """
    return np.sqrt(2*np.log(1+t) / n_pulls)


