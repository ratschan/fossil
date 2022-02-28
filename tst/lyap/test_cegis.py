# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
# 
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree. 
 
import unittest
import torch
import timeit
from src.shared.components.cegis import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.cegis_values import CegisConfig
from src.shared.consts import VerifierType, TimeDomain, CertificateType
from functools import partial
from z3 import *
from src.translator.translator_continuous import TranslatorContinuous
from src.shared.cegis_values import CegisStateKeys


def zero_in_zero(learner):
    v_zero, _ = learner.forward_tensors(
        torch.zeros(1, learner.input_size).reshape(1, learner.input_size),
    )
    return v_zero == .0


def positive_definite(learner, S_d, Sdot):
    v, _, _ = learner.forward(S_d, Sdot)
    return all(v >= .0)


def negative_definite_lie_derivative(learner, S, Sdot):
    v, vdot, _  = learner.forward(S, Sdot)
    # find points have vdot > 0
    if len(torch.nonzero(vdot > 0)) > 0:
        return False
    else:
        return True


class test_cegis(unittest.TestCase):

    def assertNumericallyLyapunov(self, model, S_d, Sdot):
        self.assertTrue(zero_in_zero(model), "model is not zero in zero")
        self.assertTrue(positive_definite(model, S_d, Sdot),
                        "model is not positive definite")
        self.assertTrue(negative_definite_lie_derivative(model, S_d, Sdot),
                        "model lie derivative is not negative definite")

    def assertLyapunovOverPositiveOrthant(self, system, c):

        f, domain, _, _ = system(functions=c.verifier.solver_fncts(),
                              inner=c.inner, outer=c.outer)
        domain = domain[0]({}, list(c.x_map.values()))
        translator = TranslatorContinuous(c.learner, np.array(c.x).reshape(-1,1), np.array(c.xdot).reshape(-1,1),
                                  None, 3)
        res = translator.get(**{'factors': None})
        V, Vdot = res[CegisStateKeys.V], res[CegisStateKeys.V_dot]

        s = Solver()
        s.add(Vdot > 0)
        s.add(domain)
        res = s.check()
        if res == z3.sat:
            model = "{}".format(s.model())
        else:
            model = ""
        self.assertEqual(res, z3.unsat, "Formally not lyapunov. Here is a cex : {}".format(model))

        Sdot = c.f_learner(c.S[0].T)
        S, Sdot = c.S[0], torch.stack(Sdot).T
        self.assertNumericallyLyapunov(c.learner, S, Sdot)

    def test_poly_2(self):
        torch.manual_seed(167)
    
        batch_size = 500
        benchmark = poly_2
        n_vars = 2
        system = partial(benchmark, batch_size)

        # define domain constraints
        outer_radius = 10
        inner_radius = 0.01

        # define NN parameters
        activations = [ActivationType.SQUARE]
        n_hidden_neurons = [10] * len(activations)

        opts = {
            CegisConfig.N_VARS.k: n_vars,
            CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
            CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
            CegisConfig.VERIFIER.k: VerifierType.Z3,
            CegisConfig.ACTIVATION.k: activations,
            CegisConfig.SYSTEM.k: system,
            CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
            CegisConfig.SP_HANDLE.k: False,
            CegisConfig.INNER_RADIUS.k: inner_radius,
            CegisConfig.OUTER_RADIUS.k: outer_radius,
            CegisConfig.LLO.k: True,
        }

        start = timeit.default_timer()
        c = Cegis(**opts)
        c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(system, c)
        
    def test_non_poly_0(self):
        torch.manual_seed(167)
        batch_size = 500
        benchmark = nonpoly0_lyap
        n_vars = 2
        system = partial(benchmark, batch_size)

        # define domain constraints
        outer_radius = 10
        inner_radius = 0.01

        # define NN parameters
        activations = [ActivationType.SQUARE]
        n_hidden_neurons = [2] * len(activations)

        start = timeit.default_timer()
        opts = {
            CegisConfig.N_VARS.k: n_vars,
            CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
            CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
            CegisConfig.VERIFIER.k: VerifierType.Z3,
            CegisConfig.ACTIVATION.k: activations,
            CegisConfig.SYSTEM.k: system,
            CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
            CegisConfig.SP_HANDLE.k: False,
            CegisConfig.INNER_RADIUS.k: inner_radius,
            CegisConfig.OUTER_RADIUS.k: outer_radius,
            CegisConfig.LLO.k: True,

        }
        c = Cegis(**opts)
        c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(system, c)

    def test_non_poly_1(self):
        torch.manual_seed(167)
        batch_size = 500
        benchmark = nonpoly1
        n_vars = 2
        system = partial(benchmark, batch_size)

        # define domain constraints
        outer_radius = 10
        inner_radius = 0.01

        # define NN parameters
        activations = [ActivationType.LINEAR, ActivationType.SQUARE]
        n_hidden_neurons = [20] * len(activations)

        opts = {
            CegisConfig.N_VARS.k: n_vars,
            CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
            CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
            CegisConfig.VERIFIER.k: VerifierType.Z3,
            CegisConfig.ACTIVATION.k: activations,
            CegisConfig.SYSTEM.k: system,
            CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
            CegisConfig.SP_HANDLE.k: False,
            CegisConfig.INNER_RADIUS.k: inner_radius,
            CegisConfig.OUTER_RADIUS.k: outer_radius,
            CegisConfig.LLO.k: True,
        }
        start = timeit.default_timer()
        c = Cegis(**opts)
        state, vars, f_learner, iters = c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(system, c)

    def test_non_poly_2(self):
        torch.manual_seed(167)
        batch_size = 750
        benchmark = nonpoly2
        n_vars = 3
        system = partial(benchmark, batch_size)

        # define domain constraints
        outer_radius = 10
        inner_radius = 0.01
        
        # define NN parameters
        activations = [ActivationType.LINEAR, ActivationType.SQUARE]
        n_hidden_neurons = [10] * len(activations)

        start = timeit.default_timer()

        opts = {
            CegisConfig.N_VARS.k: n_vars,
            CegisConfig.CERTIFICATE.k: CertificateType.LYAPUNOV,
            CegisConfig.TIME_DOMAIN.k: TimeDomain.CONTINUOUS,
            CegisConfig.VERIFIER.k: VerifierType.Z3,
            CegisConfig.ACTIVATION.k: activations,
            CegisConfig.SYSTEM.k: system,
            CegisConfig.N_HIDDEN_NEURONS.k: n_hidden_neurons,
            CegisConfig.SP_HANDLE.k: False,
            CegisConfig.INNER_RADIUS.k: inner_radius,
            CegisConfig.OUTER_RADIUS.k: outer_radius,
            CegisConfig.LLO.k: True,
        }
        c = Cegis(**opts)
        state, vars, f_learner, iters = c.solve()
        stop = timeit.default_timer()

        self.assertLyapunovOverPositiveOrthant(system, c)


if __name__ == '__main__':
    unittest.main() 
