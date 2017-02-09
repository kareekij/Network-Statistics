from __future__ import division, print_function

import networkx as nx
import argparse
import _mylib
import numpy as np

def plot_cc_vs_deg(cc,deg):

	bins = {}

	for node, node_deg in deg.iteritems():
		if node_deg not in bins:
			bins[node_deg] = [cc[node]]
		else:
			bins[node_deg].append(cc[node])

	for k,v in bins.iteritems():
		bins[k] = np.mean(np.array(bins[k]))

	_mylib.scatterPlot(bins.keys(),bins.values(),save=True,xlabels='degree',ylabels='avg. clustering coeff.',title="Facebook-combined")


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('fname', help='Edgelist file', type=str)

	args = parser.parse_args()
	fname = args.fname

	G = nx.read_edgelist(fname)
	degree = G.degree()
	cc = nx.clustering(G)

	_mylib.draw_graph_tool(G)

	# plot_cc_vs_deg(cc,degree)
	# _mylib.degreeHist(degree.values())