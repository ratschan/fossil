# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
from typing import Any

import sympy as sp
import re
from matplotlib import pyplot as plt

from experiments.benchmarks.domain_fcns import *
import experiments.benchmarks.models as models


###############################
# NON POLY BENCHMARKS
###############################

# this series comes from
# 2014, Finding Non-Polynomial Positive Invariants and Lyapunov Functions for
# Polynomial Systems through Darboux Polynomials.

# also from CDC 2011, Parrillo, poly system w non-poly lyap

def nonpoly0_lyap():
    p = models.NonPoly0()
    domain = Torus([0,0], 10, 0.1)

    return p, {'lie-&-pos': domain.generate_domain}, {'lie-&-pos':domain.generate_data(1000)}, inf_bounds_n(2)


def nonpoly0_rws():
    p = models.NonPoly0()
    XD = Sphere([0,0], 10)
    goal = Sphere([0,0], 0.1)
    unsafe = Sphere([3,3], 0.5)
    init = Sphere([-3, -3], 0.5)
    batch_size = 500
    domains = {'lie': XD.generate_domain,
                'init': init.generate_domain,
                'unsafe': unsafe.generate_boundary,
                'goal':goal.generate_domain}

    data = {'lie': SetMinus(XD, goal).generate_data(batch_size), 
            'init': init.generate_data(batch_size),
            'unsafe': unsafe.generate_data(batch_size)}

    return p, domains, data, inf_bounds_n(2)


def nonpoly1():

    outer = 10.
    batch_size = 500

    f = models.NonPoly1()

    XD = PositiveOrthantSphere([0., 0.], outer)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def nonpoly2():

    outer = 10.
    batch_size = 750

    f = models.NonPoly2()

    XD = PositiveOrthantSphere([0., 0., 0.], outer)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


def nonpoly3():

    outer = 10.
    batch_size = 500

    f = models.NonPoly3()

    XD = PositiveOrthantSphere([0., 0., 0.], outer)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


# POLY benchmarks

def benchmark_0():

    outer = 10.
    batch_size = 1000
    # test function, not to be included
    f = models.Benchmark0()

    XD = Sphere([0., 0.], outer)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def poly_1():

    outer = 10.
    inner = 0.1
    batch_size = 500
    # SOSDEMO2
    # from http://sysos.eng.ox.ac.uk/sostools/sostools.pdf
    f = models.Poly1()

    XD = Torus([0., 0., 0.], outer, inner)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(3)


# this series comes from
# https://www.cs.colorado.edu/~srirams/papers/nolcos13.pdf
# srirams paper from 2013 (old-ish) but plenty of lyap fcns
def poly_2():

    outer = 10.
    inner = 0.01
    batch_size = 500

    f = models.Poly2()

    XD = Torus([0., 0.], outer, inner)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def poly_3():

    outer = 10.
    inner = 0.1
    batch_size = 500

    f = models.Poly3()

    XD = Torus([0., 0.], outer, inner)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def poly_4():

    outer = 10.
    inner = 0.1
    batch_size = 500

    f = models.Poly4()

    XD = Torus([0., 0.], outer, inner)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def twod_hybrid():

    outer = 10.
    inner = 0.01
    batch_size = 1000
    # example of 2-d hybrid sys
    f = models.TwoDHybrid()

    XD = Torus([0., 0.], outer, inner)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def linear_discrete():

    outer = 10.
    inner = 0.01
    batch_size = 500

    f = models.LinearDiscrete()

    XD = Torus([0., 0.], outer, inner)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)


def double_linear_discrete():

    outer = 10.
    batch_size = 1000
    f = models.DoubleLinearDiscrete()

    XD = Sphere([0., 0., 0., 0.], outer)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(4)


def linear_discrete_n_vars(smt_verification, n_vars):

    outer = 10.
    batch_size = 1000

    f = models.LinearDiscreteNVars()

    XD = Sphere([0.] * n_vars, outer)

    data = {
        'lie-&-pos': XD.generate_data(batch_size)
    }

    if smt_verification:
        domains = {
            'lie-&-pos': XD.generate_domain
        }
    else:
        lower_inputs = -outer * np.ones((1, n_vars))
        upper_inputs = outer * np.ones((1, n_vars))
        initial_bound = jax_verify.IntervalBound(lower_inputs, upper_inputs)
        domains = {
            'lie-&-pos': initial_bound
        }

    return f, domains, data, inf_bounds_n(n_vars)


def non_linear_discrete():

    outer = 10.
    batch_size = 1000

    f = models.NonLinearDiscrete()

    XD = Sphere([0., 0.], outer)

    domains = {
        'lie-&-pos': XD.generate_domain,
    }

    data = {
        'lie-&-pos': XD.generate_data(batch_size),
    }

    return f, domains, data, inf_bounds_n(2)



def max_degree_fx(fx):
    return max(max_degree_poly(f) for f in fx)


def max_degree_poly(p):
    s = str(p)
    s = re.sub(r'x\d+', 'x', s)
    try:
        f = sp.sympify(s)
        return sp.degree(f)
    except:
        print("Exception in %s for %s" % (max_degree_poly.__name__, p))
        return 0
