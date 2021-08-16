import numpy as np
from scipy import signal


class Generators:

    @staticmethod
    def sine(period, amplitude=1, v_shift=0, p_shift=0):
        return lambda x: v_shift + amplitude * np.sin(x * (2*np.pi) / period + p_shift)

    @staticmethod
    def square(period, amplitude=1, duty=0.5, v_shift=0, p_shift=0):
        return lambda x: v_shift + amplitude * signal.square(x * (2*np.pi) / period + p_shift, duty=duty)

    @staticmethod
    def sawtooth(period, amplitude=1, width=0.5, v_shift=0, p_shift=0):
        return lambda x: v_shift + amplitude * signal.sawtooth(x * (2*np.pi) / period + p_shift, width=width)
