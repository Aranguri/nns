import numpy as np

class Adam:
    def __init__(self, init_ws, lr=1e-2, b1=.9, b2=.999):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.i = 1
        self.mws, self.vws = init_ws(), init_ws()

    def update(self, ws, dws):
        for w, dw, mw, vw in zip(ws, dws, self.mws, self.vws):
            mw = self.b1 * mw + (1 - self.b1) * dw
            vw = self.b2 * vw + (1 - self.b2) * dw ** 2
            mw = mw / (1 - self.b1 ** (self.i + 1))
            vw = vw / (1 - self.b2 ** (self.i + 1))
            w -= self.lr * mw / (np.sqrt(vw) + 1e-8)

        self.i += 1
        return ws
