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

from functools import reduce
import numpy as np

def by_one_iterations_algo(directory):
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

    dcs, tenants_all = from_directory(last_iteration_path)

    tenants_placed = [x for x in tenants_all if x .mark is not None]
    # tenants_to_place = tenants_all[800+100*last_iteration:800+100*(last_iteration+1)]
    tenants_to_place = tenants_all[800:]
    tenants_in_process = tenants_placed + tenants_to_place

    for tenant in tenants_in_process:
        tenant.evaluation = e.get_tenant_evaluation(tenant)

    dict_dcs = {dc.name : dc for dc in dcs}
    dict_tenants = {tenant.name : tenant for tenant in tenants_in_process}

    all_timings = []
    all_sendings = []
    all_returns = []
    is_placed = []

    start_time = time.time()
    for i in range(len(tenants_to_place)):
        all_placed, timings, sendings, returns_count = algo_step(
            dcs_per_tenant,
            dcs,
            tenants_placed+[tenants_to_place[i]],
            e,
            strategy_choose_tenant,
            strategy_choose_dc,
            dict_dcs,
            dict_tenants
        )
        all_timings.append(timings)
        all_sendings.append(sendings)
        all_returns.append(returns_count)
        is_placed.append(all_placed)

    elapsed_time = time.time() - start_time

    returns_count = reduce(lambda x, y: {"yes": x["yes"] + y["yes"], "no": x["no"] + y["no"]}, all_returns, {'no': 0, 'yes': 0})
    timings = np.array(all_timings).sum(axis=0).tolist()
    # sendings = np.array(all_sendings).T.tolist()[0]
    sendings = [[] for _ in range(8)]
    for iteration in all_sendings:
        for i, val in enumerate(iteration):
            sendings[i].extend(val)

    new_iteration_path = os.path.join(directory, str(last_iteration + 1))
    os.mkdir(new_iteration_path)
    into_directory(dcs, tenants_all, new_iteration_path)

    filename_timings = os.path.join(directory, "timings.csv")
    local_time = ",".join([str(x) if x is not None else "" for x in timings])
    line = "{},{},{}\n".format(last_iteration+1, elapsed_time, local_time)
    with open(filename_timings, "a") as f:
        f.write(line)

    filename_returns = os.path.join(directory, "returns.csv")
    line = "{},{},{}\n".format(last_iteration+1, returns_count["yes"], returns_count["no"])
    with open(filename_returns, "a") as f:
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
    all_placed, timings, sendings, returns_count = algo_step(
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

    filename_returns = os.path.join(directory, "returns.csv")
    line = "{},{},{}\n".format(last_iteration+1, returns_count["yes"], returns_count["no"])
    with open(filename_returns, "a") as f:
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
    # , cols=["локальные", "мета"]
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

def iterations_view_returned(directory):
    filename =  os.path.join(directory, "returns.csv")
    df = pd.read_csv(filename, names=["iteration", "переназначены", "не переназначены"])
    df = df.set_index("iteration")
    ax = df.plot(kind='bar', stacked=True)
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("Возвращаемых запросов")

    df_placed, _ = iterations_view_sent(directory)
    df_placed = df_placed.query("mode=='placed'")

    last_iteration = int(df.iloc[-1].name)
    for i in range(last_iteration):
        return_ok = df.iloc[i, 0]
        return_fail = df.iloc[i, 1]
        total_placed = df_placed.query("iteration=={}".format(i+1))["count"].sum()
        print("итерация", i+1)
        print("размещено за итерацию", total_placed)
        print("возвращалось за итерацию", return_ok+return_fail)
        print("из возвращаемых переназначены", return_ok)
        print("-----")

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
