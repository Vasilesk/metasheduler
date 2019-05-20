from data_loader import from_directory, into_directory
from evaluators import Evaluator_sum, Evaluator_naive
from algo import main_algo

from strategies import get_tenants_random, get_dcs_random

from strategies import get_dcs_utilized, get_dcs_emptiest
from strategies import get_tenants_heaviest, get_tenants_lightest

from checking_results import utilization_dcs, placement_tenants

import time

if __name__ == '__main__':
    dcs, tenants = from_directory("examples/dcs_empty_sorted")

    e = Evaluator_sum()

    start_time = time.time()
    main_algo(3, 1, dcs, tenants, e, get_tenants_heaviest, get_dcs_utilized)
    elapsed_time = time.time() - start_time
    print("time elapsed", int(elapsed_time), "s")

    tp = placement_tenants(tenants)
    print("tenants: {}/{}".format(tp["placed"], tp["total"]))
    ud = utilization_dcs(dcs)
    print("utilization: \navg {}\nmax {}".format(ud["avg"], ud["max"]))

    into_directory(dcs, tenants, "examples/result_dcs_empty_sorted")
