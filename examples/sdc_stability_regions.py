# coding=utf-8
import warnings as warnings
# emmit all warnings

warnings.simplefilter('always')
# Deactivate Annoyances
#  DeprecationWarnings are emitted by various numpy functions
warnings.simplefilter('ignore', category=DeprecationWarning)
#  RuntimeWarnings are emitted by numpy.abs on most calls when encountering over- or underflows.
warnings.simplefilter('ignore', category=RuntimeWarning)

import argparse
import concurrent.futures
import pickle as pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pypint.solvers.sdc import Sdc
from pypint.solvers.cores.semi_implicit_sdc_core import SemiImplicitSdcCore
from pypint.utilities.threshold_check import ThresholdCheck
from examples.problems.lambda_u import LambdaU


def run_problem(real, imag, max_iter, num_steps, num_nodes, criteria, task, num_tasks, n_procs, starttime):
    _width = len(str(num_tasks))
    _percent = float(task) / float(num_tasks) * 100
    _diff_time = time.time() - starttime
    _time_epsilon = 0.1
    if task > n_procs \
            and ((_diff_time > 8.0 and
                  ((_diff_time % 10.0) < _time_epsilon) or ((10.0 - (_diff_time % 10.0)) < _time_epsilon))
                 or (num_tasks % 2 == 0 and _percent % 4 == 0)
                 or (num_tasks % 2 != 0 and _percent % 4 == 0)):
        print("[ {:6.2f}%] Starting task {:{width}d} of {:{width}d}: \\lambda = {: .3f}{:+.3f}i"
              .format(_percent, task, num_tasks, real, imag, width=_width))

    problem = LambdaU(lmbda=complex(real, imag))
    check = ThresholdCheck(min_threshold=1e-7, max_threshold=max_iter,
                           conditions=(criteria, 'solution reduction', 'error reduction', 'iterations'))
    solver = Sdc()
    solver.init(problem=problem, threshold=check, num_time_steps=num_steps, num_nodes=num_nodes)
    try:
        solution = solver.run(SemiImplicitSdcCore)
        return int(solution.used_iterations)
    except RuntimeError:
        return max_iter + 1


def sdc_stability_region(num_points, max_iter, num_steps, num_nodes, num_procs, real, imag, criteria):
    _start_time = time.time()
    _test_region = {
        'real': real,
        'imag': imag
    }
    _dist = [
        np.abs(_test_region['real'][1] - _test_region['real'][0]),
        np.abs(_test_region['imag'][1] - _test_region['imag'][0]),
    ]
    _num_points_per_axis = {
        'real': num_points,
        'imag': num_points
    }
    if _dist[0] > _dist[1]:
        _num_points_per_axis['imag'] = int(_dist[1] / _dist[0] * num_points)
    else:
        _num_points_per_axis['real'] = int(_dist[0] / _dist[1] * num_points)

    _points = {
        'real': np.linspace(_test_region['real'][0], _test_region['real'][1], _num_points_per_axis['real']),
        'imag': np.linspace(_test_region['imag'][0], _test_region['imag'][1], _num_points_per_axis['imag'])
    }
    _results = np.zeros((_num_points_per_axis['imag'], _num_points_per_axis['real']), dtype=np.int)
    _futures = np.zeros((_num_points_per_axis['imag'], _num_points_per_axis['real']), dtype=object)

    _name = "stability_{:.2f}-{:.2f}_{:.2f}-{:.2f}_p{:d}_maxI{:d}_T{:d}_n{:d}"\
            .format(_test_region["real"][0], _test_region['real'][1], _test_region['imag'][0], _test_region['imag'][1],
                    num_points, max_iter, num_steps, num_nodes)

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_procs) as pool:
        for a in range(0, _points['real'].size):
            for j in range(0, _points['imag'].size):
                _futures[j][a] = \
                    pool.submit(run_problem, _points['real'][a], _points['imag'][j], max_iter, num_steps, num_nodes,
                                criteria, a * _points['imag'].size + j + 1, _points['real'].size * _points['imag'].size,
                                num_procs, _start_time)

    for a in range(0, _points['real'].size):
        for j in range(0, _points['imag'].size):
            if _futures[j][a].exception(timeout=None) is None:
                _results[j][a] = _futures[j][a].result(timeout=None)
            else:
                _results[j][a] = max_iter
                print("[FAILED  ] \\lambda = {: .3f}{:+.3f}i.\n[  reason] {:s}"
                      .format(_points['real'][a], _points['imag'][j], _futures[j][a].exception()))

    with open("{:s}.pickle".format(_name), 'wb') as f:
        pickle.dump(_results, f)

    plt.hold(True)
    plt.title("SDC with {:d} time steps and {:d} nodes each".format(num_steps, num_nodes))
    C = plt.contour(_points['real'], _points['imag'], _results, levels=[4, 8, 16, 32, 64, 128, 256, 512])
    plt.clabel(C, inline=1, fontsize=10, fmt='%3i')
    CF = plt.pcolor(_points['real'], _points['imag'], np.log2(_results), cmap=cm.jet, rasterized=True)
    plt.xlabel('Real part')
    plt.ylabel('Imaginary part')
    plt.grid('off')
    plt.tight_layout()
    fig = plt.gcf()
    fig.savefig("{:s}.png".format(_name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SDC Stability Anaylsis")
    parser.add_argument('-p', '--num-pnts', nargs='?', default=10, type=int, help="Number of points on longest axis.")
    parser.add_argument('-i', '--max-iter', nargs='?', default=769, type=int, help="Maximum number of iterations.")
    parser.add_argument('-t', '--num-stps', nargs='?', default=1, type=int, help="Number of time steps.")
    parser.add_argument('-n', '--num-ndes', nargs='?', default=5, type=int, help="Number of integration nodes per time step.")
    parser.add_argument('-w', '--num-proc', nargs='?', default=8, type=int, help="Number of concurrent worker processes.")
    parser.add_argument('--real', nargs=2, default=[-8.0, 2.0], type=float, help="Start and end of real axis.")
    parser.add_argument('--imag', nargs=2, default=[0.0, 7.0], type=float, help="Start and end of imaginary axis.")
    parser.add_argument('-c', '--criteria', nargs='?', default='error', type=str, help="Termination criteria.")
    args = parser.parse_args()

    print("[        ] Calculating SDC Stability Regions")
    for key in vars(args):
        print("[{:{fill}{align}8s}] {}".format(key[0:8], vars(args)[key], fill=' ', align='<'))

    sdc_stability_region(args.num_pnts, args.max_iter, args.num_stps, args.num_ndes, args.num_proc, args.real,
                         args.imag, args.criteria)
