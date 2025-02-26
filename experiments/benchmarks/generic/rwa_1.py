# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pylint: disable=not-callable
from experiments.benchmarks import models
from fossil import domains
from fossil import certificate
from fossil import main, control
from fossil.constraints import negative_nonstrict, positive_boundary, safe_progress
from fossil.consts import *

def test_lnn(args):
    n_vars = 2

    system = models.Linear1LQR
    batch_size = 500

    XD = domains.Rectangle([-1.5, -1.5], [1.5, 1.5])
    XS = domains.Rectangle([-1, -1], [1, 1])
    XI = domains.Rectangle([-0.5, -0.5], [0.5, 0.5])
    XG = domains.Rectangle([-0.1, -0.1], [0.1, 0.1])

    SU = domains.SetMinus(XD, XS)  # Data for unsafe set

    
#    sets = {
#        certificate.XD: XD,
#        certificate.XI: XI,
#        certificate.XS_BORDER: XS,
#        certificate.XS: XS,
#        certificate.XG: XG,
#    }
#    data = {
#        certificate.XD: XD._generate_data(batch_size),
#        certificate.XI: XI._generate_data(100),
#        certificate.XU: SU._generate_data(1000),
#    }

    sets = {
        "I": XI,
        "U": SU,
        "D": XD
    }

    
    data = {
        "I": XI._generate_data(100),
        "U": SU._generate_data(1000),
        "D": XD._generate_data(batch_size)  # this is what the original example uses, although this is larger than the set the constraint needs to be enforced on
    }
    
    constraints = {
        "I": negative_nonstrict,
        "U": positive_boundary,   # the loss function is calculated on the whole set, with verification only on the boundary
        "D": safe_progress
    }


    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [4] * len(activations)

    opts = CegisConfig(
        DOMAINS=sets,
        DATA=data,
        SYSTEM=system,
        N_VARS=n_vars,
        CERTIFICATE=CertificateType.GENERIC,                # emulating RWS
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=n_hidden_neurons,
        CEGIS_MAX_ITERS=25,
        CONSTRAINTS=constraints                
    )

    main.run_benchmark(
        opts,
        record=args.record,
        plot=args.plot,
        concurrent=args.concurrent,
        repeat=args.repeat,
    )


if __name__ == "__main__":
    args = main.parse_benchmark_args()
    test_lnn(args)
