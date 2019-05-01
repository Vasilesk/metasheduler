from itertools import chain
# Tenant
class Tenant:
    def __init__(self, bs):
        # self.name = ""
        # self.bs = BeautifulSoup("", "xml")
        # self.evaluation = {"":0}
        # self.mark = ""

        self.bs = bs
        self.evaluation = None

        self.name = bs["name"]

        self.mark = None

    def __str__(self):
        return 'tenant {}'.format(self.name)

    def delete_placement(self, dc, e):
        # self.bs = tenant_bs_empty
        resources = chain(
            self.bs.find_all("vm"),
            self.bs.find_all("st"),
            self.bs.find_all("port")
        )
        for vres in resources:
            del vres["assignedTo"]

        self.mark = None
        self.evaluation = e.get_tenant_evaluation(self)
        dc.tenants_placed = [x for x in dc.tenants_placed if x.name != self.name]
        dc.evaluation = e.get_dc_evaluation(dc)

    def set_placement(self, tenant_bs, chosen_dc, e):
        self.bs = tenant_bs
        self.mark = chosen_dc.name
        self.evaluation = e.get_tenant_evaluation(self)
        chosen_dc.tenants_placed.append(self)
        chosen_dc.evaluation = e.get_dc_evaluation(chosen_dc)
