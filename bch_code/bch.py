import numpy as np
import gf

class BCH:
    primpoly_list = [0, 0, 7, 11, 19, 55, 109, 193, 425]
    pm = None
    g = None
    R = None
    m = None
    n = None
    k = None
    t = None
    
    def __init__(self, n, t):
        q = int(np.log2(n + 1))
        
        assert 2 <= q <= 8, 'Wrong code word length n. Must be in [3, 2^8 - 1]'
        assert 2 ** q - 1 == n, 'Wrong code word length n. Must be 2^q - 1'
        assert 2 * t + 1 <= n, 'Number of errors t is too big'
        
        self.n = n
        self.t = t
        self.pm = gf.gen_pow_matrix(self.primpoly_list[q])
        self.R = self.pm[: 2 * t, 1]
        self.g, _ = gf.minpoly(self.R, self.pm)
        
        assert ((self.g == 0) | (self.g == 1)).all()
        p = np.zeros(n + 1).astype(int)
        p[0] = 1
        p[-1] = 1
        assert (gf.polydivmod(p, self.g, self.pm)[1] == 0).all()
        
        self.m = self.g.size - 1
        self.k = self.n - self.m
        
    def encode(self, U):
        if U.ndim == 1:
            U = U[np.newaxis, :]
        assert U.shape[1] == self.k
        V = np.zeros((U.shape[0], self.n)).astype(int)
        x_pow_m = np.zeros(self.m + 1).astype(int)
        x_pow_m[0] = 1
        for i, word in enumerate(U):
            xu = gf.polyprod(word, x_pow_m, self.pm)
            _, mod = gf.polydivmod(xu, self.g, self.pm)
            code = gf.polyadd(xu, mod)
            code = np.concatenate([np.zeros(self.n - code.size).astype(int), code])
            
            assert (gf.polydivmod(code, self.g, self.pm)[1] == 0).all()
            assert (gf.polyval(code, self.R, self.pm) == 0).all()
            V[i] = code
        return V
    
    def decode(self, W, method='euclid'):
        if W.ndim == 1:
            W = W[np.newaxis, :]
        decoded = np.zeros_like(W).astype(int)
        for i, w in enumerate(W):
            s = gf.polyval(w, self.R, self.pm)
            if (s == 0).all():
                decoded[i] = w
                continue
            if method == 'pgz':
                for nu in range(self.t, 0, -1):
                    A = np.array([[s[k] for k in range(j, nu + j)] for j in range(nu)])
                    b = [s[j] for j in range(nu, 2 * nu)]
                    Lambda = gf.linsolve(A, b, self.pm)
                    if Lambda is not np.nan:
                        break
                if Lambda is np.nan:
                    decoded[i] = np.full(self.n, np.nan)
                    continue
                Lambda = np.append(Lambda, 1)
            elif method == 'euclid':
                z = np.zeros(2 * self.t + 2).astype(int)
                z[0] = 1
                S = np.ones(2 * self.t + 1).astype(int)
                S[:-1] = s[::-1]
                _, _, Lambda = gf.euclid(z, S, self.pm, max_deg=self.t)
                
            vals = gf.polyval(Lambda, self.pm[:,1], self.pm)
            indices = np.array([j for j in range(vals.size)])
            zero_idx = indices[vals == 0]
            roots = self.pm[zero_idx, 1]
            if roots.size != 0:
                err_idx = self.pm[gf.divide(np.full(roots.size, 1), roots, self.pm) - 1, 0]
                w[self.n - 1 - err_idx] = w[self.n - 1 - err_idx] ^ 1
            
            if method == 'pgz':
                s_new = gf.polyval(w, self.R, self.pm)
                if (s_new != 0).any():
                    w = np.full(self.n, np.nan)
            if method == 'euclid' and roots.size != Lambda.size - 1:
                w = np.full(self.n, np.nan)
                
            decoded[i] = w
        return decoded

    def dist(self):
        d = np.inf
        for x in range(1, 2**self.k):
            input_msg = np.array([int(i) for i in bin(x)[2:]])
            input_msg = np.concatenate([np.zeros(self.k - input_msg.size), input_msg]).astype(int)
            d = min((self.encode(input_msg) != 0).sum(), d)
        return d    