# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks import models
from src import domains
from src import certificate
from src import main
from src.consts import *


def test_lnn():
    outer = 1
    inner = 0.1
    batch_size = 1500
    open_loop = models.InvertedPendulum

    XD = domains.Torus([0.0, 0.0], outer, inner)

    system = models.GeneralClosedLoopModel.prepare_from_open(open_loop())

    sets = {
        certificate.XD: XD,
    }
    data = {
        certificate.XD: XD._generate_data(batch_size),
    }

    # TEST for Control Lyapunov
    # pass the ctrl parameters from here (i.e. the main)
    n_vars = 2

    # define NN parameters
    lyap_activations = [ActivationType.SQUARE]
    lyap_hidden_neurons = [5] * len(lyap_activations)

    # ctrl params
    n_ctrl_inputs = 2

    opts = CegisConfig(
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.LYAPUNOV,
        LLO=False,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=lyap_activations,
        N_HIDDEN_NEURONS=lyap_hidden_neurons,
        CTRLAYER=[25, n_ctrl_inputs],
        CTRLACTIVATION=[ActivationType.LINEAR],
    )
    main.run_benchmark(opts, record=False, plot=True, repeat=1)


if __name__ == "__main__":
    torch.manual_seed(167)
    test_lnn()