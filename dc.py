from bs4 import BeautifulSoup
import time

from tenant import Tenant

# Datacenter
class DC:
    def __init__(self, bs):
        # self.name = ""
        # self.bs = BeautifulSoup("", "xml")
        # self.evaluation = {"":0}
        # self.tenants_placed = [Tenant(bs)]
        # self.names_rejected = set("")
        # self.tenants_try = [Tenant(bs)]

        self.bs = bs
        self.name = bs.find("dcxml")["name"]

        self.evaluation = None

        self.tenants_placed = []

        self.names_rejected = set()

        self.tenants_try = []

        for tenant_bs in bs.find_all("tenant"):
            assigned = all(
                [(x.get("assignedTo", None) is not None) for x in (list(tenant_bs.find_all("vm")) + list(tenant_bs.find_all("st")))]
            )
            if assigned:
                tenant = Tenant(tenant_bs)
                self.tenants_placed.append(tenant)
                tenant.mark = self.name

        bs.find("tenants").extract()
        bs.find("dcxml").append(bs.new_tag('tenants'))

    def __str__(self):
        return 'datacenter {}'.format(self.name)

    def get_xml(self):
        result = BeautifulSoup(self.bs.prettify(), "xml")
        result_tenants = result.find("tenants")

        for tenant in self.tenants_placed + self.tenants_try:
            result_tenants.append(tenant.bs)

        return result.prettify()

    def parse_output(self, filename):
        placed = []
        not_placed = []
        removes = dict()

        content = None
        with open(filename, "r") as f:
            content = f.read()
        bs = BeautifulSoup(content, 'xml')

        for tenant_bs in bs.find_all("tenant"):
            checker = lambda x: (x.get("assignedTo", None) is not None)
            assigned_data = [checker(x) for x in (list(tenant_bs.find_all("vm")) + list(tenant_bs.find_all("st")))]

            assigned = any(assigned_data)
            if assigned != all(assigned_data):
                print("PARTIAL ASSIGNING ERROR")

            if assigned:
                removed_because = tenant_bs.get("removedBecause", None)
                if removed_because is not None:
                    caused_removes = removes.get(removed_because, [])
                    caused_removes.append(tenant_bs["name"])
                    removes[removed_because] = caused_removes

                placed.append(tenant_bs)
            else:
                not_placed.append(tenant_bs)

        placed = [(x, removes.get(x["name"], [])) for x in placed]

        return placed, not_placed

    def pre_exec(self):
        if len(self.tenants_try) == 0:
            return None

        xml = self.get_xml()

        filename_in = "./tmp/in_dc_{}_{}.xml".format(self.name, int(time.time()))
        filename_out = "./tmp/out_dc_{}_{}.xml".format(self.name, int(time.time()))
        with open(filename_in, "w") as f:
            f.write(xml)

        return filename_in, filename_out



    def after_exec(self, filename_out):
        if filename_out is None:
            return []

        placed, not_placed = self.parse_output(filename_out)

        new_rejects = set([x["name"] for x in not_placed])
        self.names_rejected |= new_rejects

        tries = set([x.name for x in self.tenants_try])
        new_possible = [x for x in placed if x[0]["name"] in tries]

        self.tenants_try = []

        return new_possible
