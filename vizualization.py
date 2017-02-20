from __future__ import division, print_function
import sys
from graph_tool.all import *
import csv
import random
import _mylib
import argparse
import networkx as nx

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('fname', help='Edgelist file', type=str)
    parser.add_argument('outname', help='output name', type=str)

    args = parser.parse_args()
    fname = args.fname
    outname = args.outname

    G = nx.read_edgelist(fname)
    LCC = max(nx.connected_component_subgraphs(G), key=len)

    _mylib.draw_graph_tool(LCC,outname)
