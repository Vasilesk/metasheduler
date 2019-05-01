from bs4 import BeautifulSoup
import os

from dc import DC
from tenant import Tenant

from evaluators import Evaluator_base

def read_tenants_from_file(filename):
    tenants = []
    content = None
    with open(filename, "r") as f:
        content = f.read()
    bs = BeautifulSoup(content, 'xml')
    for tenant_bs in bs.find_all("tenant"):
        tenants.append(Tenant(tenant_bs))

    return tenants

def tenants_gen1():
    return read_tenants_from_file("../data/merged_tenants.xml")
    # tenants = []
    # content = None
    # with open("../data/merged_tenants.xml", "r") as f:
    #     content = f.read()
    # bs = BeautifulSoup(content, 'xml')
    # for tenant_bs in bs.find_all("tenant"):
    #     tenants.append(Tenant(tenant_bs))
    #
    # return tenants

def dcs_gen1():
    paths = ["../data/generated_dcs/{}.xml".format(x) for x in range(1, 21)]
    dcs = []
    for path in paths:
        with open(path, "r") as f:
            content = f.read()
        bs = BeautifulSoup(content, 'xml')
        bs.find("tenants").extract()
        bs.find("dcxml").append(bs.new_tag('tenants'))

        dcs.append(DC(bs))

    return(dcs)

def tenants_gen2():
    return read_tenants_from_file("../data/merged_tenants_all.xml")
    # tenants = []
    # content = None
    # with open("../data/merged_tenants_all.xml", "r") as f:
    #     content = f.read()
    # bs = BeautifulSoup(content, 'xml')
    # for tenant_bs in bs.find_all("tenant"):
    #     tenants.append(Tenant(tenant_bs))
    #
    # return tenants

def dcs_gen2():

    return(dcs_gen1()[:8])

def domestic():
    filenames = os.listdir("../data/domestic_dcs/")
    paths = ["../data/domestic_dcs/{}".format(x) for x in filenames]
    paths = ["../data/domestic_dcs/{}.xml".format(x) for x in range(1,4)]
    dcs = []
    for path in paths:
        with open(path, "r") as f:
            content = f.read()
        bs = BeautifulSoup(content, 'xml')
        bs.find("tenants").extract()
        bs.find("dcxml").append(bs.new_tag('tenants'))

        dcs.append(DC(bs))

    tenants = []
    content = None
    with open("../data/domestic_tenants.xml", "r") as f:
        content = f.read()
    bs = BeautifulSoup(content, 'xml')
    for tenant_bs in bs.find_all("tenant"):
        tenants.append(Tenant(tenant_bs))

    return dcs, tenants

def with_return():
    paths = ["../data/domestic_return/{}.xml".format(x) for x in range(1,3)]
    dcs = []
    for path in paths:
        with open(path, "r") as f:
            content = f.read()
        bs = BeautifulSoup(content, 'xml')
        bs.find("tenants").extract()
        bs.find("dcxml").append(bs.new_tag('tenants'))

        dcs.append(DC(bs))

    tenants = []
    content = None
    with open("../data/domestic_return/tenants.xml", "r") as f:
        content = f.read()
    bs = BeautifulSoup(content, 'xml')
    for tenant_bs in bs.find_all("tenant"):
        tenants.append(Tenant(tenant_bs))

    dcs[0].tenants_placed.append(tenants[0])
    tenants[0].mark = "fst"

    return dcs, tenants

def from_directory(directory):
    dcs = []
    tenants = []

    filenames = os.listdir(directory)
    dcs_filenames = [x for x in filenames if x != "tenants.xml"]

    dcs_filenames = sorted(dcs_filenames, key=lambda x: int(x.split(".")[0]))

    dcs_paths = [os.path.join(directory, x) for x in dcs_filenames]
    for path in dcs_paths:
        content = None
        with open(path, "r") as f:
            content = f.read()

        bs = BeautifulSoup(content, 'xml')
        dc = DC(bs)
        dcs.append(dc)
        tenants.extend(dc.tenants_placed)


    path = os.path.join(directory, "tenants.xml")
    tenants += read_tenants_from_file(path)

    # return dcs, tenants, tenants_placed
    e = Evaluator_base()
    for tenant in tenants:
        tenant.evaluation = e.get_tenant_evaluation(tenant)

    return dcs, tenants

def into_directory(dcs, tenants, directory):
    for dc in dcs:
        path = os.path.join(directory, "{}.xml".format(dc.name))
        with open(path, "w") as f:
            content = dc.get_xml()
            f.write(content)

    tenants_remained = [x.bs.prettify() for x in tenants if x.mark is None]
    template = '''<?xml version="1.0" encoding="utf-8"?>
<tenants>
{}
</tenants>
'''
    out_data = template.format("\n".join(tenants_remained))
    path = os.path.join(directory, "tenants.xml")
    with open(path, "w") as f:
        f.write(out_data)


def folder_loader(template_folder_xml, dcs_count):
    dcs = []
    # tenants_placed = []

    for i in range(1, dcs_count+1):
        path = template_folder_xml.format(i)
        content = None
        with open(path, "r") as f:
            content = f.read()

        bs = BeautifulSoup(content, 'xml')
        dcs.append(DC(bs))
        # for tenant_bs in bs.find_all("tenant"):
        #     tenants_placed.append(Tenant(tenant_bs))

    path = template_folder_xml.format("remained_tenants")

    tenants = read_tenants_from_file(path)

    # return dcs, tenants, tenants_placed
    return dcs, tenants


def dcs_storer(dcs, tenants, template_folder_xml):
    for i, dc in enumerate(dcs):
        xml = dc.get_xml()
        path = template_folder_xml.format(i+1)
        with open(path, "w") as f:
            f.write(xml)

    tenants_not_placed = [x for x in tenants if x.mark is None]
    remained_tenants = [x.bs.prettify() for x in tenants_not_placed]
    template = '''<?xml version="1.0" encoding="utf-8"?>
<tenants>
{}
</tenants>
'''
    out_data = template.format("\n".join(remained_tenants))
    with open(template_folder_xml.format("remained_tenants"), "w") as f:
        f.write(out_data)
