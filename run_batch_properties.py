from __future__ import division, print_function

import networkx as nx
import argparse
import _mylib
import numpy as np
import community
import os
import pickle
import csv


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

def find_communities(dataset, G):
	com_fname = './pickle/p_{}.pickle'.format(dataset)

	if os.path.isfile(com_fname):
		partition = pickle.load(open(com_fname, 'rb'))
	else:
		partition = community.best_partition(G)
		pickle.dump(partition, open(com_fname, 'wb'))

	mod = community.modularity(partition, G)
	return partition, mod

def community_size(G,p):
	p_size = {}
	for idx in set(p.values()):
		members = _mylib.get_members_from_com(idx, p)
		p_size[idx] = len(members)

	return p_size

def read_check_file(folder_path):
	fname = folder_path + 'check.txt'

	files_check = set()

	if not os.path.isfile(fname):
		return files_check

	file = open(fname, "r")
	print("Reading check.txt .. ")
	for line in (file.readlines()):
		line = line.replace("\n","")
		print(	' {} '.format(line))
		files_check.add(line)
	print('-'*10)
	return files_check

def update_check_file(folder_path, filename):
	fname = folder_path + 'check.txt'
	file = open(fname, "a")
	file.write(filename+'\n')
	file.close()

def write_to_file(fname,data):
	if not os.path.isfile(fname):
		with open(fname, 'w') as fp:
			a = csv.writer(fp, delimiter=',')
			a.writerows([['filename','nodes','edges','avg.deg','avg.cc','avg.com_size', 'min.com_size','max.com_size', 'mod','com_count']])

	with open(fname, 'a') as fp:
		a = csv.writer(fp, delimiter=',')
		a.writerows(data)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('folder', help='Folder name', type=str)
	#
	args = parser.parse_args()
	folder = args.folder#'data3'

	folder_path = os.getcwd() + '/' + folder + '/'

	print('Folder:', folder_path)
	files_checker = read_check_file(folder_path)


	for filename in os.listdir(folder_path):
		data = []
		if filename == 'check.txt':
			continue

		if filename not in files_checker:
			row = []
			print('Checking File: ', filename)


			#graph = nx.read_edgelist(folder_path + filename)
			#filename = 'rt_gop.edges'
			graph = _mylib.read_file(folder_path + filename)
			print(nx.info(graph))
			G = max(nx.connected_component_subgraphs(graph), key=len)

			# Calculate different properties
			nodes_count = G.number_of_nodes()
			edges_count = G.number_of_edges()

			deg = G.degree().values()
			avg_deg = np.mean(np.array(deg))
			avg_cc = nx.average_clustering(G)

			p, mod = find_communities(filename, G)
			com_count = len(set(p.values()))
			com_size = community_size(G, p)
			avg_com_size = np.mean(np.array(com_size.values()))
			min_com_size = min(com_size.values())
			max_com_size = max(com_size.values())


			row.append(filename)

			row.append(nodes_count)
			row.append(edges_count)
			row.append(avg_deg)
			row.append(avg_cc)
			row.append(avg_com_size)
			row.append(min_com_size)
			row.append(max_com_size)
			row.append(mod)
			row.append(com_count)

			data.append(row)

			# print('	Avg. Deg:', avg_deg)
			# print(' Avg. CC:', avg_cc)
			# print(' Q:', mod)
			# print(' Avg. Com size:', avg_com_size)
			update_check_file(folder_path, filename)
			print(" 		.. checked !".format(filename))
		else:
			print("	 - {} PASSED ".format(filename))


		write_to_file('./stats.txt',data)




#
# LCC = max(nx.connected_component_subgraphs(G), key=len)
