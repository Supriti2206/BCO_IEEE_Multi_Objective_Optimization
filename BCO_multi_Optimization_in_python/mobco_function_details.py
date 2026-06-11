"""
Multi-Objective Benchmark Functions for N Objectives
"""

import numpy as np

def mobco_func_details(fname, n_obj=2):
    """Returns objective function, bounds, and dimension"""
    
    if fname == 'zdt1':
        if n_obj != 2:
            print(f"⚠️ ZDT1 only supports 2 objectives! Using DTLZ1 with {n_obj} objectives instead.")
            return mobco_func_details('dtlz1', n_obj)
        fobj = lambda x: zdt1(x)
        lb = 0
        ub = 1
        dim = 30
        
    elif fname == 'zdt2':
        if n_obj != 2:
            print(f"⚠️ ZDT2 only supports 2 objectives! Using DTLZ2 with {n_obj} objectives instead.")
            return mobco_func_details('dtlz2', n_obj)
        fobj = lambda x: zdt2(x)
        lb = 0
        ub = 1
        dim = 30
        
    elif fname == 'zdt3':
        if n_obj != 2:
            print(f"⚠️ ZDT3 only supports 2 objectives! Using DTLZ1 with {n_obj} objectives instead.")
            return mobco_func_details('dtlz1', n_obj)
        fobj = lambda x: zdt3(x)
        lb = 0
        ub = 1
        dim = 30
        
    elif fname == 'zdt4':
        if n_obj != 2:
            print(f"⚠️ ZDT4 only supports 2 objectives! Using DTLZ1 with {n_obj} objectives instead.")
            return mobco_func_details('dtlz1', n_obj)
        fobj = lambda x: zdt4(x)
        lb = 0
        ub = 1
        dim = 10
        
    elif fname == 'zdt6':
        if n_obj != 2:
            print(f"⚠️ ZDT6 only supports 2 objectives! Using DTLZ2 with {n_obj} objectives instead.")
            return mobco_func_details('dtlz2', n_obj)
        fobj = lambda x: zdt6(x)
        lb = 0
        ub = 1
        dim = 10
        
    elif fname == 'dtlz1':
        fobj = lambda x: dtlz1(x, n_obj)
        lb = 0
        ub = 1
        dim = n_obj + 9
        
    elif fname == 'dtlz2':
        fobj = lambda x: dtlz2(x, n_obj)
        lb = 0
        ub = 1
        dim = n_obj + 9
        
    elif fname == 'dtlz3':
        fobj = lambda x: dtlz3(x, n_obj)
        lb = 0
        ub = 1
        dim = n_obj + 9
        
    elif fname == 'dtlz4':
        fobj = lambda x: dtlz4(x, n_obj)
        lb = 0
        ub = 1
        dim = n_obj + 9
        
    else:
        raise ValueError(f"Unknown function: {fname}")
    
    return fobj, lb, ub, dim


# ZDT Functions (2 objectives only)
def zdt1(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])


def zdt2(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - (f1 / g)**2
    f2 = g * h
    return np.array([f1, f2])


def zdt3(x):
    n = len(x)
    f1 = x[0]
    g = 1 + 9 * np.sum(x[1:]) / (n - 1)
    h = 1 - np.sqrt(f1 / g) - (f1 / g) * np.sin(10 * np.pi * f1)
    f2 = g * h
    return np.array([f1, f2])


def zdt4(x):
    n = len(x)
    f1 = x[0]
    g_sum = 0
    for i in range(1, n):
        g_sum += x[i]**2 - 10 * np.cos(4 * np.pi * x[i])
    g = 1 + 10 * (n - 1) + g_sum
    h = 1 - np.sqrt(f1 / g)
    f2 = g * h
    return np.array([f1, f2])


def zdt6(x):
    n = len(x)
    f1 = 1 - np.exp(-4 * x[0]) * np.sin(6 * np.pi * x[0])**6
    g_sum = 0
    for i in range(1, n):
        g_sum += x[i]
    g = 1 + 9 * (g_sum / (n - 1))**0.25
    h = 1 - (f1 / g)**2
    f2 = g * h
    return np.array([f1, f2])


# DTLZ Functions (ANY number of objectives!)
def dtlz1(x, M):
    n = len(x)
    k = n - M + 1
    
    g_sum = 0
    for i in range(n - k, n):
        g_sum += (x[i] - 0.5)**2 - np.cos(20 * np.pi * (x[i] - 0.5))
    g = 100 * (k + g_sum)
    
    f = np.zeros(M)
    product = 1.0
    for i in range(M - 1):
        product *= x[i]
        f[i] = 0.5 * product * (1 + g)
    f[M - 1] = 0.5 * (1 - x[0]) * (1 + g)
    
    return f


def dtlz2(x, M):
    n = len(x)
    k = n - M + 1
    
    g_sum = 0
    for i in range(n - k, n):
        g_sum += (x[i] - 0.5)**2
    g = g_sum
    
    f = np.zeros(M)
    product = 1.0
    for i in range(M - 1):
        product *= np.cos(x[i] * np.pi / 2)
        f[i] = (1 + g) * product
    f[M - 1] = (1 + g) * np.sin(x[0] * np.pi / 2)
    
    return f


def dtlz3(x, M):
    n = len(x)
    k = n - M + 1
    
    g_sum = 0
    for i in range(n - k, n):
        g_sum += (x[i] - 0.5)**2 - np.cos(20 * np.pi * (x[i] - 0.5))
    g = 100 * (k + g_sum)
    
    f = np.zeros(M)
    product = 1.0
    for i in range(M - 1):
        product *= np.cos(x[i] * np.pi / 2)
        f[i] = (1 + g) * product
    f[M - 1] = (1 + g) * np.sin(x[0] * np.pi / 2)
    
    return f


def dtlz4(x, M):
    alpha = 100
    n = len(x)
    k = n - M + 1
    
    g_sum = 0
    for i in range(n - k, n):
        g_sum += (x[i] - 0.5)**2
    g = g_sum
    
    f = np.zeros(M)
    product = 1.0
    for i in range(M - 1):
        product *= np.cos((x[i]**alpha) * np.pi / 2)
        f[i] = (1 + g) * product
    f[M - 1] = (1 + g) * np.sin((x[0]**alpha) * np.pi / 2)
    
    return f