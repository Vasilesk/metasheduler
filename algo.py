from multiprocessing import Pool
from subprocess import call, check_output
import io

def scheduler_exec(data):
    if data is None:
        return None

    filename_in, filename_out = data

    f_prog = open("tmp/out_prog","wb")

    exec_path = "local_algo/yulia_algo"
    # exec_path = "local_algo/andrei_algo"

    call_data = [
        exec_path,
        filename_in,
        filename_out,
        "/dev/null",
    ]

    call(call_data, stdout=f_prog, stderr=f_prog)

    f_prog.close()
    return filename_out

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

    tenants_processed = [x for x in tenants if x.mark is None]

    while len(tenants_processed) > 0 and iter_max > 0:
        iter_max -= 1

        tenant_placements = dict()

        for dc in dcs:
            dc.evaluation = e.get_dc_evaluation(dc)

        tenants_prior = strategy_choose_tenant(tenants_processed)
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
        with Pool(8) as p:
            filenames_out = p.map(scheduler_exec, exec_input)
        for dc, filename in zip(dcs, filenames_out):
            placements = dc.after_exec(filename)
            for placement, removings in placements:
                tenant_placements[placement["name"]].append((dc.name, placement, removings))

        # for the next step
        tenants_processed = []

        # choose dc for tenant
        for name in tenant_placements:
            tenant = dict_tenants[name]
            placement_chosen = choose_placement(
                tenant,
                tenant_placements[name],
                dict_dcs,
                dict_tenants,
                strategy_choose_dc,
                e,
                strategy_choose_tenant
            )

            if not placement_chosen:
                tenants_processed.append(tenant)

    everything_placed = all([x.mark is not None for x in tenants])
    return iter_max, everything_placed

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
        # # no looking
        # return False

        if delta_conditional == 0:
            return False

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

        _, placed_ok = main_algo(
            1,
            len(other_dcs),
            other_dcs,
            tenants_condition,
            e,
            strategy_choose_tenant,
            strategy_choose_dc
        )

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
