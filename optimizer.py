import numpy as np
from utils import *
from copy import deepcopy

class Adam:
    def __init__(self, init_ws, lr=1e-1, b1=.9, b2=.999):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.i = 1
        self.mws, self.vws = init_ws(), init_ws()

    def update(self, ws, dws):
        for j, (w, dw, mw, vw) in enumerate(zip(ws, dws, self.mws, self.vws)):
            mw = self.b1 * mw + (1 - self.b1) * dw
            vw = self.b2 * vw + (1 - self.b2) * dw ** 2
            mw = mw / (1 - self.b1 ** (self.i + 1))
            vw = vw / (1 - self.b2 ** (self.i + 1))
            w -= self.lr * mw / (vw ** (1/2) + 1e-8)
            self.mws[j], self.vws[j] = mw, vw
        self.i += 1
        return ws

class SGD:
    def __init__(self, lr=1e-2):
        self.lr = lr

    def update(self, ws, dws):
        for w, dw in zip(ws, dws):
            w -= self.lr * dw
        return ws

class Adagrad:
    def __init__(self, init_ws, lr=1e-1):
        self.lr = lr
        self.mws = init_ws()

    def update(self, ws, dws):
        #wrong implementation self.mws doesn't change
        for w, dw, mw in zip(ws, dws, self.mws):
            mw += dw ** 2
            w -= self.lr * dw / ((mw ** (1/2) + 1e-12))
        return ws

class Momentum:
    def __init__(self, init_ws, lr=1e-2, friction=.9):
        self.vws = init_ws()
        self.lr = lr
        self.friction = friction

    def update(self, ws, dws):
        for i, (w, dw) in enumerate(zip(ws, dws)):
            if i == 0:
                new = dw.shape[1] - self.vws[i].shape[1]
                self.vws[i] = np.pad(self.vws[i], [(0, 0), (0, new)], mode='constant')
            self.vws[i] = self.friction * self.vws[i] + dw
            w -= self.lr * self.vws[i]
        return ws
