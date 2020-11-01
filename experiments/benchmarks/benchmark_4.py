import torch
import timeit
from src.lyap.cegis_lyap import Cegis
from experiments.benchmarks.benchmarks_lyap import *
from src.shared.activations import ActivationType
from src.shared.consts import VerifierType, LearnerType, TrajectoriserType, RegulariserType
from src.shared.cegis_values import CegisConfig, CegisStateKeys
from src.plots.plot_lyap import plot_lyce
from functools import partial


def test_lnn():
    batch_size = 500
    benchmark = benchmark_4
    n_vars = 2
    system = partial(benchmark, batch_size)

    # define domain constraints
    outer_radius = 10
    inner_radius = 0.01

    # define NN parameters
    activations = [ActivationType.SQUARE]
    n_hidden_neurons = [5] * len(activations)

    opts = {
        CegisConfig.N_VARS.k: n_vars,
        CegisConfig.LEARNER.k: LearnerType.NN,
        CegisConfig.VERIFIER.k: VerifierType.DREAL,
        CegisConfig.TRAJECTORISER.k: TrajectoriserType.DEFAULT,
        CegisConfig.REGULARISER.k: RegulariserType.DEFAULT,
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
    print('Elapsed Time: {}'.format(stop - start))

    # plotting -- only for 2-d systems
    if len(vars) == 2 and state[CegisStateKeys.found]:
        plot_lyce(np.array(vars), state[CegisStateKeys.V],
                      state[CegisStateKeys.V_dot], f_learner)


if __name__ == '__main__':
    torch.manual_seed(167)
    test_lnn()