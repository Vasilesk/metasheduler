import json

from collections import Counter
from evaluators import Evaluator_sum

def dump_experiment(benchmarks, filename):
    with open(filename, "w") as f:
        json.dump(benchmarks, f)

def load_experiment(filename):
    with open(filename, "r") as f:
        result = json.load(f)

    return result

# def utilization_resources(elem, base):
#     utilizations = []
#     for key in ["RAM", "VCPUs", "storage"]:
#         absolute = base[key] - elem[key]
#
#         try:
#             div_val = absolute / base[key]
#         except:
#             div_val = 0
#         utilizations.append(div_val)
#
#     return max(utilizations)

def utilization_dcs(dcs):
    e = Evaluator_sum()

    evals_base = [e.get_empty_dc_evaluation(x)["VCPUs"] for x in dcs]

    evals = [y - e.get_dc_evaluation(x)["VCPUs"] for x, y in zip(dcs, evals_base)]

    divs = [x/y if y !=0 else 0 for x, y in zip(evals, evals_base)]


    # utilization_values = []
    # for dc, dc_base in zip(evals, evals_base):
    #     value = utilization_resources(dc, dc_base)
    #     utilization_values.append(value)
    #
    return {
        "avg": sum(evals) / sum(evals_base),
        "max": max(divs),
        "vals": divs,
    }

def placement_tenants(tenants):
    c = Counter()
    for tenant in tenants:
        if tenant.mark is not None:
            c[tenant.mark] += 1
    vals = []
    for k, v in c.items():
        vals.append((k, v))

    return {
        "placed": sum(c.values()),
        "total": len(tenants),
        "vals": vals,
    }
