from multiprocessing import Pool
from subprocess import call, check_output
import io
import time

from checking_results import utilization_dcs, placement_tenants

def scheduler_exec(data):
    if data is None:
        return None

    filename_in, filename_out = data

    f_prog = open("tmp/out_prog","wb")

    exec_path = "local_algo/9_algo"
    # exec_path = "local_algo/true_andrei_algo"

    call_data = [
        exec_path,
        filename_in,
        filename_out,
        "/dev/null",
    ]

    start_time = time.time()

    call(call_data, stdout=f_prog, stderr=f_prog)
    f_prog.close()

    elapsed_time = time.time() - start_time
    return elapsed_time

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def batched_algo(
        batch_size,
        iter_max,
        dcs_per_tenant,
        dcs,
        tenants,
        e,
        strategy_choose_tenant,
        strategy_choose_dc
):
    batches = batch(tenants, batch_size)
    batch_results = []

    for batch_tenants in batches:
        batch_result = main_algo(iter_max, dcs_per_tenant, dcs, tenants, e, strategy_choose_tenant, strategy_choose_dc)
        batch_results.append(batch_result)

    return batch_results

def main_algo(
        iter_max,
        dcs_per_tenant,
        dcs,
        tenants,
        e,
        strategy_choose_tenant,
        strategy_choose_dc
):
    for tenant in tenants:
        tenant.evaluation = e.get_tenant_evaluation(tenant)

    dict_dcs = {dc.name : dc for dc in dcs}
    dict_tenants = {tenant.name : tenant for tenant in tenants}

    benchmarks = []
    tenants_processed = [x for x in tenants if x.mark is None]
    while len(tenants_processed) > 0 and iter_max > 0:
        start_time = time.time()
        _, timings = algo_step(
            dcs_per_tenant,
            dcs,
            tenants_processed,
            e,
            strategy_choose_tenant,
            strategy_choose_dc,
            dict_dcs,
            dict_tenants
        )

        elapsed_time = time.time() - start_time
        benchmark = (placement_tenants(tenants), utilization_dcs(dcs), timings, elapsed_time)
        benchmarks.append(benchmark)

        iter_max -= 1
        tenants_processed = [x for x in tenants if x.mark is None]

    everything_placed = all([x.mark is not None for x in tenants])
    return benchmarks, everything_placed

def algo_step(
    dcs_per_tenant,
    dcs,
    tenants,
    e,
    strategy_choose_tenant,
    strategy_choose_dc,
    dict_dcs,
    dict_tenants
):
    tenant_placements = dict()

    for dc in dcs:
        dc.evaluation = e.get_dc_evaluation(dc)

    tenants_prior = strategy_choose_tenant(tenants)
    for tenant in tenants_prior:
        dcs_count = 0
        for dc in strategy_choose_dc(tenant, dcs):
            if tenant.name in dc.names_rejected:
                continue
            evaluation, placement_possible = e.place(dc, tenant)
            if placement_possible:
                dc.evaluation = evaluation
                dc.tenants_try.append(tenant)

                dcs_count += 1
                if dcs_count == dcs_per_tenant:
                    break
        if dcs_count != 0:
            tenant_placements[tenant.name] = []


    exec_input = [x.pre_exec() for x in dcs]
    filenames_out = [x[1] if x is not None else None for x in exec_input]
    with Pool(8) as p:
        timings = p.map(scheduler_exec, exec_input)
    for dc, filename in zip(dcs, filenames_out):
        placements = dc.after_exec(filename)
        for placement, removings in placements:
            tenant_placements[placement["name"]].append((dc.name, placement, removings))

    # choose dc for tenant
    for name in tenant_placements:
        tenant = dict_tenants[name]
        choose_placement(
            tenant,
            tenant_placements[name],
            dict_dcs,
            dict_tenants,
            strategy_choose_dc,
            e,
            strategy_choose_tenant
        )

    return all([x.mark is not None for x in tenants]), timings

def choose_placement(tenant, possible_placements, dict_dcs, dict_tenants, strategy_choose_dc, e, strategy_choose_tenant):
    if len(possible_placements) == 0:
        return False

    possible_dcs_names_unconditional = [x[0] for x in possible_placements if len(x[2]) == 0]
    possible_dcs_unconditional = [dict_dcs[x] for x in possible_dcs_names_unconditional]

    delta_conditional = len(possible_placements) - len(possible_dcs_names_unconditional)
    if delta_conditional != 0:
        print("dcs proposing taking away:", delta_conditional)

    if len(possible_dcs_unconditional) > 0:
        chosen_dc = strategy_choose_dc(tenant, possible_dcs_unconditional)[0]
        chosen_placement = [x[1] for x in possible_placements if x[0] == chosen_dc.name][0]

        tenant.set_placement(chosen_placement, chosen_dc, e)
        # tenant.bs = chosen_placement
        # tenant.mark = chosen_dc.name
        # tenant.evaluation = e.get_tenant_evaluation(tenant)
        # chosen_dc.tenants_placed.append(tenant)
        # chosen_dc.evaluation = e.get_dc_evaluation(chosen_dc)

        return True

    else:
        # # for no looking
        # return False

        if delta_conditional == 0:
            return False

        print("ACCEPTED")

        possible_dcs_names_conditional = [x[0] for x in possible_placements if len(x[2]) != 0]
        # possible_dcs_conditional = [dict_dcs[x] for x in possible_dcs_names_conditional]
        possible_conditions = [x[2] for x in possible_placements if len(x[2]) != 0]

        # mb will be changed
        tenant_names_condition = possible_conditions[0]
        chosen_dc = dict_dcs[possible_dcs_names_conditional[0]]
        chosen_placement = [x[1] for x in possible_placements if len(x[2]) != 0][0]

        other_dcs = [dict_dcs[x] for x in dict_dcs if x not in possible_dcs_names_conditional]

        print("recursive search for {} tenants".format(len(tenant_names_condition)))
        tenants_condition = [dict_tenants[x] for x in tenant_names_condition]
        placement_backup = [(x.bs, dict_dcs[x.mark]) for x in tenants_condition]


        delete_placements(tenants_condition, dict_dcs, e)

        placed_ok, timings = algo_step(
            len(other_dcs),
            other_dcs,
            tenants_condition,
            e,
            strategy_choose_tenant,
            strategy_choose_dc,
            dict_dcs,
            dict_tenants
        )

        # TODO: timings usage

        if placed_ok:
            tenant.set_placement(chosen_placement, chosen_dc, e)
        else:
            set_placements(tenants_condition, placement_backup, e)

        return placed_ok

def set_placements(tenants, placements, e):
    for tenant, placement in zip(tenants, placements):
        bs, ds = placement
        tenant.set_placement(bs, ds, e)

def delete_placements(tenants, dict_dcs, e):
    for tenant in tenants:
        if tenant.mark is not None:
            dc = dict_dcs[tenant.mark]
            tenant.delete_placement(dc, e)
