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

def sign_loss_neg(V, Vdot, circle):  
  return V

def sign_verif_neg(connectives, variables, V, Vdot):
  return V>=0

def sign_loss_pos(V, Vdot, circle):  
  return V

def sign_verif_pos(connectives, variables, V, Vdot):
  return V<=0

def barrier_loss_belt(B_d, Bdot_d, circle):            
        margin = 0
        belt_index = torch.nonzero(torch.abs(B_d) <= 0.5)

        if belt_index.nelement() != 0:
            dB_belt = torch.index_select(Bdot_d, dim=0, index=belt_index[:, 0])          # choose the elements of Bdot_d for which B_d is close to zero
            loss= (torch.relu(dB_belt + 0 * margin)).mean()
            percent_belt = (
                100 * ((dB_belt <= -margin).count_nonzero()).item() / dB_belt.shape[0]
            )            
        else:
            loss= 0
            percent_belt= 0
          
        return loss, percent_belt

def barrier_verif(connectives, variables, V, Vdot):
        return _And(V == 0, Vdot >= 0)

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
        certificate.XD: XD,
        certificate.XI: XI,
        certificate.XU: XU,
    }
    data = {
        certificate.XD: XD._generate_data(500),
        certificate.XI: XI._generate_data(500),
        certificate.XU: XU._generate_data(500),
    }

    constraints = {
      "initial": {"loss": sign_loss_neg, "verif": sign_verif_neg},
      "unsafe": {"loss": sign_loss_pos, "verif": sign_verif_pos},
      "inductivity": {"loss": barrier_loss_belt, "verif": barrier_verif }
    }

    
    system = models.Barr1
    activations = [ActivationType.SIGMOID]
    hidden_neurons = [5] * len(activations)
    opts = CegisConfig(
        N_VARS=2,
        SYSTEM=system,
        DOMAINS=sets,
        DATA=data,
        CERTIFICATE=CertificateType.BARRIER,
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
