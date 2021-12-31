#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
import tensorflow as tf
from tensorflow import keras
import tet

class RLModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
