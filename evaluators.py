class Evaluator_base:
    def __init__(self):
        pass

    def get_dc_evaluation(self, dc):
        raise NotImplementedError("should be implemented")

    # returns new evaluation and flag if placement is possible
    def place(self, dc, tenant):
        raise NotImplementedError("should be implemented")

    def get_tenant_evaluation(self, tenant):
        return self.get_tenant_dict_from_tenant_bs(tenant.bs)

    def get_vms_from_tenant_bs(self, bs):
        vms = []
        for vm in bs.find("list_of_nodes").find_all("vm"):
            bs_params = vm.find("parameter_set")
            vm_cpu = bs_params.find("parameter", {"parameter_name": "VCPUs"})
            vm_ram = bs_params.find("parameter", {"parameter_name": "RAM"})
            vms.append({
                "RAM": int(vm_ram["parameter_value"]),
                "VCPUs": int(vm_cpu["parameter_value"]),
                "assignedTo": vm.get("assignedTo", None),
            })
        return vms

    def get_sts_from_tenant_bs(self, bs):
        sts = []
        for st in bs.find("list_of_nodes").find_all("st"):
            st_size = st.find("parameter_set").find("parameter", {"parameter_name": "size"})
            sts.append({
                "size": int(st_size["parameter_value"]),
                "assignedTo": st.get("assignedTo", None),
            })
        return sts

    def get_tenant_dict_from_tenant_bs(self, bs):
        vms = self.get_vms_from_tenant_bs(bs)
        sts = self.get_sts_from_tenant_bs(bs)

        return {
            "sts": sts,
            "vms": vms,
        }

    def get_server_from_bs(self, bs):
        params = bs.find("parameter_set")
        return {
            "name": bs["name"],
            "RAM": int(params.find("parameter", {"parameter_name": "RAM"})["parameter_value"]),
            "VCPUs": int(params.find("parameter", {"parameter_name": "VCPUs"})["parameter_value"]),
        }

    def get_storage_from_bs(self, bs):
        params = bs.find("parameter_set")
        return {
            "name": bs["name"],
            "size": int(params.find("parameter", {"parameter_name": "size"})["parameter_value"]),
        }

    def get_dc_servers_storages(self, dc):
        servers = []
        storages = []

        resources = dc.bs.find("resources")
        for resource in resources.find_all("server"):
            server = self.get_server_from_bs(resource)
            servers.append(server)

        for resource in resources.find_all("storage"):
            storage = self.get_storage_from_bs(resource)
            storages.append(storage)

        return servers, storages

    def reduce_resources_by_tenants(self, servers, storages, tenants):
        dict_servers = {x["name"]: x for x in servers}
        dict_storages = {x["name"]: x for x in storages}

        for tenant in tenants:
            tenant_dict = tenant.evaluation

            for st in tenant_dict["sts"]:
                assigned = st["assignedTo"]
                if assigned is not None:
                    dict_storages[assigned]["size"] -= st["size"]

            for vm in tenant_dict["vms"]:
                assigned = vm["assignedTo"]
                if assigned is not None:
                    dict_servers[assigned]["RAM"] -= vm["RAM"]
                    dict_servers[assigned]["VCPUs"] -= vm["VCPUs"]

    def dc_summed_data(self, servers, storages):
        return {
            "RAM": sum((x["RAM"] for x in servers)),
            "VCPUs": sum((x["VCPUs"] for x in servers)),
            "storage": sum((x["size"] for x in storages)),
        }

    def tenant_summed_data(self, tenant_evaluation):
        return {
            "storage": sum((x["size"] for x in tenant_evaluation["sts"])),
            "RAM": sum((x["RAM"] for x in tenant_evaluation["vms"])),
            "VCPUs": sum((x["VCPUs"] for x in tenant_evaluation["vms"])),
        }

class Evaluator_naive(Evaluator_base):
    def __init__(self):
        Evaluator_base.__init__(self)

    def get_dc_evaluation(self, dc):
        return None

    def place(self, dc, tenant):
        return None, True


class Evaluator_sum(Evaluator_base):
    def __init__(self):
        Evaluator_base.__init__(self)

    def get_dc_evaluation(self, dc):
        servers, storages = self.get_dc_servers_storages(dc)

        self.reduce_resources_by_tenants(servers, storages, dc.tenants_placed)

        return self.dc_summed_data(servers, storages)

    def get_empty_dc_evaluation(self, dc):
        servers, storages = self.get_dc_servers_storages(dc)

        return self.dc_summed_data(servers, storages)

    def place(self, dc, tenant):
        tenant_summed = self.tenant_summed_data(tenant.evaluation)

        for res in ["storage", "RAM", "VCPUs"]:
            if dc.evaluation[res] < tenant_summed[res]:
                return None, False

        for vm in tenant.evaluation["vms"]:
            for res in ["RAM", "VCPUs"]:
                if dc.evaluation[res] < vm[res]:
                    return None, False
        for st in tenant.evaluation["sts"]:
            if dc.evaluation["storage"] < st["size"]:
                return None, False


        new_evaluation = dict()
        for res in ["storage", "RAM", "VCPUs"]:
            new_evaluation[res] = dc.evaluation[res] - tenant_summed[res]

        return new_evaluation, True

class Evaluator_detailed(Evaluator_base):
    def __init__(self):
        Evaluator_base.__init__(self)

    def get_dc_evaluation(self, dc):
        servers, storages = self.get_dc_servers_storages(dc)
        self.reduce_resources_by_tenants(servers, storages, dc.tenants_placed)

        evaluation = self.dc_summed_data(servers, storages)
        evaluation["detailed"] = {"servers": servers, "storages": storages}

        return evaluation

    def possible_server_exists(self, servers, vm):
        for server in servers:
            if server["VCPUs"] >= vm["VCPUs"] and server["RAM"] >= vm["RAM"]:
                return True
        return False

    def possible_storage_exists(self, storages, st):
        for storage in storages:
            if storage["size"] >= st["size"]:
                return True
        return False

    def place(self, dc, tenant):
        tenant_summed = self.tenant_summed_data(tenant.evaluation)

        for res in ["storage", "RAM", "VCPUs"]:
            if dc.evaluation[res] < tenant_summed[res]:
                return None, False

        for vm in tenant.evaluation["vms"]:
            if not self.possible_server_exists(dc.evaluation["detailed"]["servers"], vm):
                return None, False
            for res in ["RAM", "VCPUs"]:
                if dc.evaluation[res] < vm[res]:
                    return None, False
        for st in tenant.evaluation["sts"]:
            if not self.possible_storage_exists(dc.evaluation["detailed"]["storages"], st):
                return None, False
            if dc.evaluation["storage"] < st["size"]:
                return None, False


        new_evaluation = {"detailed": dc.evaluation["detailed"]}
        for res in ["storage", "RAM", "VCPUs"]:
            new_evaluation[res] = dc.evaluation[res] - tenant_summed[res]

        return new_evaluation, True
