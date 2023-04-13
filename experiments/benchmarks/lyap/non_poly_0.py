# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

# pylint: disable=not-callable
import torch

from experiments.benchmarks.benchmarks_lyap import *
from experiments.analysis import Recorder
from src.shared.components.cegis import Cegis
from src.shared.consts import *


def test_lnn():
    benchmark = nonpoly0_lyap
    X = Torus([0, 0], 1, 0.01)
    n_vars = 2
    system = benchmark

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [2] * len(activations)

    start = timeit.default_timer()
    opts = CegisConfig(
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        SYSTEM=system,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        LLO=True,
        CEGIS_MAX_ITERS=1,
        XD=X,  # This doesn't do anything, just convienient for recording
    )
    c = Cegis(opts)
    state, vars, f, iters = c.solve()
    stop = timeit.default_timer()
    elapsed = stop - start
    print("Elapsed Time: {}".format(elapsed))
    rec = Recorder()
    rec.record(opts, state, elapsed, iters)


if __name__ == "__main__":
    for i in range(10):
        torch.manual_seed(167 + i)
        torch.set_num_threads(1)
        test_lnn()
