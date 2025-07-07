from VSSM.main import VSSM
import numpy as np

class VSSMWrapper(VSSM):
    classes_ = [0,1]

    def _arr_to_d(self, x_nom):
        d = {}
        for i,x in enumerate(x_nom):
            d[i] = chr(97+x)
        return d
    def ifit(self, x_nom, y):
        d = self._arr_to_d(x_nom)
        return super().ifit(d, y==1)

    def predict(self, X_nom):
        ys = np.zeros(len(X_nom), dtype=np.int64)
        for i, x_nom in enumerate(X_nom):
            d = self._arr_to_d(x_nom)
            ys[i] = 1 if super().predict(d) else 0
        return ys

    def predict_proba(self, X_nom):
        probs = np.zeros((len(X_nom),2))
        for i, x_nom in enumerate(X_nom):
            d = self._arr_to_d(x_nom)
            p = super().score(d)
            probs[i,0] = 1.0-p
            probs[i,1] = p
        return probs

    def score(self, x_nom):
        probs = np.zeros(2)
        d = self._arr_to_d(x_nom)
        return super().score(d)



if __name__ == "__main__":
    data = np.random.randint(1,5, size=(6,3))
    vsm_w = VSSMWrapper()

    for x in data[:1]:
        print("Q")
        vsm_w.ifit(x, 1)

    for x in data[3:6]:
        print("A")
        s = vsm_w.score(x)
        ps = vsm_w.predict_proba(x)
        print("s",s, ps)




