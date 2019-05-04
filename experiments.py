from data_loader import from_directory, into_directory
from evaluators import Evaluator_sum, Evaluator_naive, Evaluator_detailed
from algo import main_algo

from strategies import get_dcs_random, get_dcs_utilized, get_dcs_emptiest
from strategies import get_tenants_random, get_tenants_heaviest, get_tenants_lightest

from checking_results import utilization_dcs, placement_tenants
from checking_results import load_experiment, dump_experiment

import os

def view_dc_utilization(s_evaluator, s_tenant_strategy, s_dc_strategy):
    import seaborn as sns
    import pandas as pd

    path = "experiments/andrei/empty800/{}/{}/{}".format(s_evaluator, s_tenant_strategy, s_dc_strategy)
    filename = os.path.join(path, "benchmark.json")
    benchmarks = load_experiment(filename)
    df = []

    for i, benchmark in enumerate(benchmarks):
        vals = benchmark[1]["vals"]
        for j,val in enumerate(vals):
            df.append({"итераций": i+1, "номер ЦОД":j+1, "загрузка ресурсов (%)": val})

    print("средняя загрузка", benchmarks[-1][1]["avg"])

    df = pd.DataFrame().from_records(df)
    return sns.barplot(x="номер ЦОД", y="загрузка ресурсов (%)", hue="итераций", data=df)

def view_tenants_placed(s_evaluator, s_tenant_strategy, s_dc_strategy):
    import seaborn as sns
    import pandas as pd

    path = "experiments/andrei/empty800/{}/{}/{}".format(s_evaluator, s_tenant_strategy, s_dc_strategy)
    filename = os.path.join(path, "benchmark.json")
    benchmarks = load_experiment(filename)
    df = []

    vals_dict = {i: [] for i in range(1, 9)}
    for i, benchmark in enumerate(benchmarks):
        vals = benchmark[0]["vals"]
        for name, value in vals:
            vals_dict[int(name)].append(value)
    for i in range(8):
        for j, value in enumerate(vals_dict[i+1]):
            df.append({"итераций": j+1, "номер ЦОД": i+1, "размещено запросов": value})

    print("размещено запросов", benchmarks[-1][0]["placed"])

    df = pd.DataFrame().from_records(df)
    return sns.barplot(x="номер ЦОД", y="размещено запросов", hue="итераций", data=df)



def empty_experiment_run(s_evaluator, s_tenant_strategy, s_dc_strategy):
    map_evaluator = {
        "naive": Evaluator_naive,
        "sum": Evaluator_sum,
        "detailed": Evaluator_detailed,
    }
    map_tenant = {
        "tenant_random": get_tenants_random,
        "tenant_heaviest": get_tenants_heaviest,
        "tenant_lightest": get_tenants_lightest,
    }
    map_dc = {
        "dc_random": get_dcs_random,
        "dc_utilized": get_dcs_utilized,
        "dc_emptiest": get_dcs_emptiest,
    }

    e = map_evaluator[s_evaluator]()

    dcs, tenants = from_directory("examples/dcs_empty_sorted/")
    tenants1, tenants2 = tenants[:800], tenants[800:]

    benchmarks, _ = main_algo(3, 1, dcs, tenants1, e, map_tenant[s_tenant_strategy], map_dc[s_dc_strategy])

    path = "experiments/andrei/empty800/{}/{}/{}".format(s_evaluator, s_tenant_strategy, s_dc_strategy)
    dump_experiment(benchmarks, os.path.join(path, "benchmark.json"))
    into_directory(dcs, tenants, os.path.join(path, "data"))
