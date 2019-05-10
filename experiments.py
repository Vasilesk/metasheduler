from data_loader import from_directory, into_directory
from evaluators import Evaluator_sum, Evaluator_naive, Evaluator_detailed
from algo import main_algo, algo_step

from strategies import get_dcs_random, get_dcs_utilized, get_dcs_emptiest
from strategies import get_tenants_random, get_tenants_heaviest, get_tenants_lightest

from checking_results import utilization_dcs, placement_tenants
from checking_results import load_experiment, dump_experiment

import os

import json
import time

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def iterations_algo(directory):
    map_evaluator = {
        "naive": Evaluator_naive,
        "sum": Evaluator_sum,
        "detailed": Evaluator_detailed,
    }
    map_tenant = {
        "random": get_tenants_random,
        "heaviest": get_tenants_heaviest,
        "lightest": get_tenants_lightest,
    }
    map_dc = {
        "random": get_dcs_random,
        "utilized": get_dcs_utilized,
        "emptiest": get_dcs_emptiest,
    }

    filename_config = os.path.join(directory, "config.json")
    with open(filename_config, "r") as f:
        config = json.load(f)

    dcs_per_tenant = config["dcs_per_tenant"]
    e = map_evaluator[config["evaluator"]]()
    strategy_choose_tenant = map_tenant[config["tenant"]]
    strategy_choose_dc = map_dc[config["dc"]]

    filenames = os.listdir(directory)
    iterations = [int(x) for x in filenames if x.isdigit()]
    last_iteration = max(iterations)
    last_iteration_path = os.path.join(directory, str(last_iteration))

    dcs, tenants = from_directory(last_iteration_path)

    for tenant in tenants:
        tenant.evaluation = e.get_tenant_evaluation(tenant)

    dict_dcs = {dc.name : dc for dc in dcs}
    dict_tenants = {tenant.name : tenant for tenant in tenants}

    start_time = time.time()
    all_placed, timings, sendings = algo_step(
        dcs_per_tenant,
        dcs,
        tenants,
        e,
        strategy_choose_tenant,
        strategy_choose_dc,
        dict_dcs,
        dict_tenants
    )
    elapsed_time = time.time() - start_time

    new_iteration_path = os.path.join(directory, str(last_iteration + 1))
    os.mkdir(new_iteration_path)
    into_directory(dcs, tenants, new_iteration_path)

    filename_timings = os.path.join(directory, "timings.csv")
    local_time = ",".join([str(x) if x is not None else "" for x in timings])
    line = "{},{},{}\n".format(last_iteration+1, elapsed_time, local_time)
    with open(filename_timings, "a") as f:
        f.write(line)

    filename_sendings = os.path.join(directory, "sendings.csv")
    acceptings = []
    for sending, dc in zip(sendings, dcs):
        accepting = set(sending) - dc.names_rejected
        acceptings.append(list(accepting))

    line = [last_iteration+1, sendings, acceptings]

    line = json.dumps(line) + "\n"

    with open(filename_sendings, "a") as f:
        f.write(line)

def iterations_view_timing(directory):
    colnames = ["суммарное время"]+[str(x+1) for x in range(8)]
    df = pd.read_csv(os.path.join(directory, "timings.csv"), names=colnames)

    df_ax = df.fillna(0).iloc[:,1:]

    ax = df_ax.T.plot(kind='bar', stacked=True)
    ax.set_xlabel("Номер ЦОД")
    ax.set_ylabel("Время работы планировщика")

    return df, ax

def iterations_view_timing_total(df):
    total = df.iloc[:, 0]
    local = df.iloc[:, 1:].T.max()
    df_main = pd.DataFrame({"локальные": local, "мета": total}).T
    df_diff = df_main.diff()
    df_diff.iloc[0] = df_main.iloc[0]
    ax = df_diff.T.plot(kind='bar', stacked=True)

    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("Время работы")

    return ax

def iterations_view_utilization(directory):
    iterations = [int(x) for x in os.listdir(directory) if x.isdigit()]
    iterations = sorted(iterations)

    data = []
    columns = [str(x+1) for x in range(8)]

    for iteration in iterations:
        dirname = os.path.join(directory, str(iteration))
        dcs, tenants = from_directory(dirname)
        benchmark = utilization_dcs(dcs)
        data.append(benchmark["vals"])

    df = pd.DataFrame(data=data, columns=columns)

    df_diff = df.diff()
    df_diff.iloc[0] = df.iloc[0]

    ax = df_diff.T.plot(kind='bar', stacked=True)
    ax.set_xlabel("Номер ЦОД")
    ax.set_ylabel("Загрузка ресурсов")

    return df, ax

def iterations_view_placed(directory):
    iterations = [int(x) for x in os.listdir(directory) if x.isdigit()]
    iterations = sorted(iterations)

    df = pd.DataFrame(columns=[str(i+1) for i in range(8)])

    for iteration in iterations:
        dirname = os.path.join(directory, str(iteration))
        dcs, tenants = from_directory(dirname)
        benchmark = placement_tenants(tenants)
        elem = {k:v for k,v in benchmark["vals"]}
        df = df.append(elem, ignore_index=True)

    df_diff = df.diff()
    df_diff.iloc[0] = df.iloc[0]

    ax = df_diff.T.plot(kind='bar', stacked=True)
    ax.set_xlabel("Номер ЦОД")
    ax.set_ylabel("Размещено запросов")

    return df, ax

def iterations_view_sent(directory):
    import pandas as pd

    # df = pd.DataFrame(columns=["sent", "placed", "iteration", "dc"])
    df = pd.DataFrame(columns=["iteration", "dc", "mode", "count"])

    with open(os.path.join(directory, "sendings.csv")) as f:
        data = f.readlines()

    data = [json.loads(x) for x in data]
    for iteration in data:
        for dc, sent, placed in zip(range(8), iteration[1], iteration[2]):
            elem = {
                "count": len(sent),
                "mode": "sent",
                "iteration": iteration[0],
                "dc": dc+1,
            }
            df = df.append(elem, ignore_index=True)

            elem = {
                "count": len(placed),
                "mode": "placed",
                "iteration": iteration[0],
                "dc": dc+1,
            }
            df = df.append(elem, ignore_index=True)


    df_counts = df.groupby(["mode", "iteration"])["count"].sum().reset_index().iloc[:, 1:]
    df_counts = pd.concat([
        df_counts.loc[0:len(df_counts)/2-1].set_index("iteration").T,
        df_counts.loc[len(df_counts)/2:].set_index("iteration").T
    ]).reset_index().iloc[:, 1:]
    df_counts.set_axis(["размещены", "не размещены"], inplace=True)
    df_counts_diff = df_counts.diff()
    df_counts_diff.iloc[0] = df_counts.iloc[0]
    ax = df_counts_diff.iloc[::-1].T.plot(kind='bar', stacked=True)
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("Отправлено запросов")

    return df, ax

def view_dc_utilization(algo, dataset, s_evaluator, s_tenant_strategy, s_dc_strategy):
    import seaborn as sns
    import pandas as pd

    path = "experiments/{}/{}/{}/{}/{}".format(algo, dataset, s_evaluator, s_tenant_strategy, s_dc_strategy)
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

def view_tenants_placed(algo, dataset, s_evaluator, s_tenant_strategy, s_dc_strategy):
    import seaborn as sns
    import pandas as pd

    path = "experiments/{}/{}/{}/{}/{}".format(algo, dataset, s_evaluator, s_tenant_strategy, s_dc_strategy)
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

def experiment_run(algo, dataset_from, store_to, tenants_used, tenants_to_use, s_evaluator, s_tenant_strategy, s_dc_strategy):
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

    dcs, tenants = from_directory(os.path.join("examples", dataset_from))
    tenants1, tenants2 = tenants[:tenants_used], tenants[tenants_used:]

    tenants_for_run = [x for x in tenants1 if x.mark is not None] + tenants2[:tenants_to_use]

    benchmarks, _ = main_algo(3, 1, dcs, tenants_for_run, e, map_tenant[s_tenant_strategy], map_dc[s_dc_strategy])

    path = "experiments/{}/{}/{}/{}/{}".format(algo, store_to, s_evaluator, s_tenant_strategy, s_dc_strategy)
    dump_experiment(benchmarks, os.path.join(path, "benchmark.json"))
    into_directory(dcs, tenants, os.path.join(path, "data"))

def empty_experiment_run(algo, s_evaluator, s_tenant_strategy, s_dc_strategy):
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

    path = "experiments/{}/empty800/{}/{}/{}".format(algo, s_evaluator, s_tenant_strategy, s_dc_strategy)
    dump_experiment(benchmarks, os.path.join(path, "benchmark.json"))
    into_directory(dcs, tenants, os.path.join(path, "data"))
