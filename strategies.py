import random
from evaluators import Evaluator_sum

def get_tenants_random(tenants):
    shuffled = tenants[:]
    random.shuffle(shuffled)
    return shuffled

def measure_tenant(tenant):
    if tenant.evaluation is None:
        e = Evaluator_sum()
        tenant.evaluation = e.get_tenant_evaluation(tenant)
    return (
        sum((x["VCPUs"] for x in tenant.evaluation["vms"])),
        sum((x["RAM"] for x in tenant.evaluation["vms"])),
        sum((x["size"] for x in tenant.evaluation["sts"])),
    )

def get_tenants_heaviest(tenants):
    return sorted(tenants, key=lambda x: measure_tenant(x), reverse=True)

def get_tenants_lightest(tenants):
    return sorted(tenants, key=lambda x: measure_tenant(x))


def get_dcs_random(tenant, dcs):
    shuffled = dcs[:]
    random.shuffle(shuffled)
    return shuffled

def measure_dc(dc):
    vcpus = dc.evaluation["VCPUs"]
    ram = dc.evaluation["RAM"]
    size = dc.evaluation["storage"]

    return (vcpus, ram, size)

def get_dcs_emptiest(tenant, dcs):
    return sorted(dcs, key=lambda x: measure_dc(x), reverse=True)

def get_dcs_utilized(tenant, dcs):
    return sorted(dcs, key=lambda x: measure_dc(x))
