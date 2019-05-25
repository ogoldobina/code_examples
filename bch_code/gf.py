import numpy as np

def gen_pow_matrix(primpoly):
    q = primpoly.bit_length() - 1
    max_deg = 2 ** q
    mat = np.empty((max_deg - 1, 2), dtype=int)
    x = 2
    for i in range(1, max_deg):
        mat[i - 1, 1] = x
        mat[x - 1, 0] = i
        x *= 2
        if x >= max_deg:
            x ^= primpoly
    return mat        

def add(X, Y):
    return np.bitwise_xor(X, Y)

def sum(X, axis=0):
    return np.bitwise_xor.reduce(X, axis=axis)

def divide(X, Y, pm):
    def elementwize_divide(x, y):
        assert y != 0, 'division by zero'
        if x == 0:
            res = 0
        else:
            res = pm[(pm[x-1, 0] - pm[y-1, 0]) % pm.shape[0] - 1 , 1]
        return res
    return np.vectorize(elementwize_divide)(X, Y)

def prod(X, Y, pm):
    def elementwize_prod(x, y):
        if x == 0 or y == 0:
            res = 0
        else:
            res = pm[(pm[x-1, 0] + pm[y-1, 0]) % pm.shape[0] - 1 , 1]
        return res
    return np.vectorize(elementwize_prod)(X, Y)

def swap_collumns(i, j, A, indices):
    tmp = A[:,i].copy()
    A[:,i] = A[:,j]
    A[:,j] = tmp
    tmp = indices[i]
    indices[i] = indices[j]
    indices[j] = tmp    

def linsolve(A, b, pm):
    B = A.copy()
    c = b.copy()
    num_rows = A.shape[0]
    indices = [i for i in range(num_rows)]
    for i in range(num_rows):
        non_zero_indices = np.nonzero(B[i:,:])
        if np.unique(non_zero_indices[0]).size != num_rows - i:
            return np.nan
        j = non_zero_indices[1][0]
        if j != i:
            swap_collumns(i, j, B, indices)
        c[i] = divide(c[i], B[i,i], pm)
        B[i] = divide(B[i], np.full_like(B[i], B[i,i]), pm)
        for j in range(i+1,num_rows):
            if B[j,i] != 0:
                c[j] = add(c[j], prod(c[i], B[j,i], pm))
                B[j] = add(B[j], prod(B[i], np.full_like(B[j], B[j,i]), pm))
        
    res = np.zeros(num_rows).astype(int)
    res[-1] = c[-1]
    for i in range(num_rows - 2, -1, -1):
        res[i] = sum(prod(res[i + 1:], B[i,i+1:], pm)) ^ c[i]
    
    res = res[indices]
    return res

def minpoly(x, pm):
    roots = set(x)
    for alpha in set(x):
        a = int(prod(alpha, alpha, pm))
        while a != alpha:
            roots.add(a)
            a = int(prod(a, a, pm))
    poly = np.array([1])
    for r in roots:
        poly = polyprod(poly, np.array([1, r]), pm)
    return poly, np.array(list(roots))

def polyval(p, x, pm):
    vals = np.zeros(x.size).astype(int)
    for i, x_elem in enumerate(x):
        x_pow = np.zeros(p.size).astype(int)
        cur_x = 1
        for j in range(p.size - 1, -1, -1):
            x_pow[j] = cur_x
            cur_x = int(prod(cur_x, x_elem, pm))
        vals[i] = sum(prod(x_pow, p, pm))
        
    return vals        

def polyprod(p1, p2, pm):
    deg_p1 = p1.size - 1
    deg_p2 = p2.size - 1
    res_deg = deg_p1 + deg_p2
    res_coef = np.zeros(res_deg + 1).astype(int)
    for i in range(deg_p1 + 1):
        for j in range(deg_p2 + 1):
            deg = i + j
            res_coef[res_deg - deg] = add(res_coef[res_deg - deg],
                                              prod(p1[deg_p1 - i],p2[deg_p2 - j], pm))
    return res_coef

def del_zeros(p):
    non_zero_idx = np.nonzero(p)[0]
    if non_zero_idx.size != 0:
        p1 = p[np.nonzero(p)[0][0]:]
    else:
        p1 = np.array([0])
    return p1

def polydivmod(p1, p2, pm):
    p1 = del_zeros(p1).copy()
    p2 = del_zeros(p2).copy()
    deg_p1 = p1.size - 1
    deg_p2 = p2.size - 1
    if deg_p1 < deg_p2:
        return np.array([0]), p1
    div = np.zeros(deg_p1 - deg_p2 + 1).astype(int)
    while deg_p1 >= deg_p2:
        coef = divide(p1[0], p2[0], pm)
        div[- deg_p1 + deg_p2 - 1] = coef
        p1 = polyadd(p1, polyprod(div[- deg_p1 + deg_p2 - 1:], p2, pm))
        if p1.size == 1 and p1[0] == 0:
            deg_p1 = -1
        else:
            deg_p1 = p1.size - 1
    return div, p1

def polyadd(p1, p2):
    if p1.size > p2.size:
        p1, p2 = p2, p1
    p1 = np.concatenate([np.zeros(p2.size - p1.size).astype(int), p1])
    return del_zeros(add(p1, p2))

def euclid(p1, p2, pm, max_deg=0):
    deg_p1 = p1.size - 1
    deg_p2 = p2.size - 1
    if deg_p2 > deg_p1:
        swap = True
        a0, a1 = p2, p1
    else:
        swap = False
        a0, a1 = p1, p2
    x0, x1 = np.array([1]), np.array([0])
    y0, y1 = np.array([0]), np.array([1])
    while a1.size - 1 > max_deg:
        q, r = polydivmod(a0, a1, pm)
        a0, a1 = a1, r
        x0, x1 = x1, polyadd(x0, polyprod(x1, q, pm))
        y0, y1 = y1, polyadd(y0, polyprod(y1, q, pm))
    if swap:
        return a1, y1, x1
    else:
        return a1, x1, y1
