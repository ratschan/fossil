# Copyright (c) 2021, Alessandro Abate, Daniele Ahmed, Alec Edwards, Mirco Giacobbe, Andrea Peruffo
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import timeit

from fossil import domains
from fossil import certificate
from fossil import main
from experiments.benchmarks import models
from fossil.consts import *
from fossil.constraints import negative_strict, positive_strict, barrier

class UnsafeDomain(domains.Set):
    dimension = 2

    def generate_domain(self, v):
        x, y = v
        return x + y**2 <= 0

    def generate_data(self, batch_size):
        points = []
        limits = [[-2, -2], [0, 2]]
        while len(points) < batch_size:
            dom = domains.square_init_data(limits, batch_size)
            idx = torch.nonzero(dom[:, 0] + dom[:, 1] ** 2 <= 0)
            points += dom[idx][:, 0, :]
        return torch.stack(points[:batch_size])


def test_lnn(args):
    XD = domains.Rectangle([-2, -2], [2, 2])
    XI = domains.Rectangle([0, 1], [1, 2])
    XU = UnsafeDomain()

    sets = {
        "initial": XI,
        "unsafe": XU,
        "inductivity": XD,        
    }
    data = {
        "initial": XI._generate_data(500),
        "unsafe": XU._generate_data(500),
        "inductivity": XD._generate_data(500),        
    }
    constraints = {
      "initial": negative_strict,  # this is not necessary to be strict, but the original example used a strict constraint, here
      "unsafe": positive_strict,
      "inductivity": barrier
    }
    
    system = models.Barr1
    activations = [ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)
    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.GENERIC,
        TIME_DOMAIN=TimeDomain.CONTINUOUS,
        VERIFIER=VerifierType.DREAL,
        ACTIVATION=activations,
        N_HIDDEN_NEURONS=hidden_neurons,
        SYMMETRIC_BELT=True,
        VERBOSE=2,
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
