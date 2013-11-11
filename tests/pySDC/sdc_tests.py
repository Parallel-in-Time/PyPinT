import unittest
import numpy as np
from pySDC.sdc import SDC


testNumTimeSteps = [1, 3]
testNumSubSteps = [2, 4]
testNumIterations = [1, 3]
testFunctions = [
    {
        'function': lambda t, phi_t: np.ones(phi_t.size) * -1.0,
        'exact': lambda t: -t + 1.0,
        'initial_value': 1.0,
        'time_range': [0.0, 1.0],
        'msg': "Linear descending function."
    }
]


def run_sdc(params):
    my_sdc = SDC()
    my_sdc.fnc = params['function']
    my_sdc.exact = params['exact']
    my_sdc.initial_value = params['initial_value']
    my_sdc.time_range = params['time_range']
    my_sdc.iterations = params['iterations']
    my_sdc.time_steps = params['time_steps']
    my_sdc.num_substeps = params['sub_steps']
    my_sdc.solve()


def test_sdc():
    for iterations in range(testNumIterations[0], testNumIterations[1] + 1):
        for time_steps in range(testNumTimeSteps[0], testNumTimeSteps[1] + 1):
            for sub_steps in range(testNumSubSteps[0], testNumSubSteps[1] + 1):
                for params in testFunctions:
                    params['iterations'] = iterations
                    params['time_steps'] = time_steps
                    params['sub_steps'] = sub_steps
                    yield run_sdc, params


class SDCTests(unittest.TestCase):
    def test_sdc_initialization(self):
        SDC()


if __name__ == "__main__":
    unittest.main()
