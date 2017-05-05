from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt
import csv

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph
from sklearn import datasets, preprocessing
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm


if __name__ == '__main__':
	X = []
	with open('./stats4.txt', 'r') as csvfile:

		csvreader = csv.reader(csvfile)

		# This skips the first row of the CSV file.
		# csvreader.next() also works in Python 2.
		next(csvreader)

		for row in csvreader:
			X.append(row)

	X = np.array(X)
	datasets = X[:, [0]]

	X_p = np.array(X[:, [3,5,8]],dtype=float)
	X = X_p

	X_normalized = preprocessing.normalize(X_p, norm='l2')
	#print(X_normalized)


	#print(X_p)
	a = [2]
	#a = [2,3,4,5,6]

	for n_clusters in a:
		# Create a subplot with 1 row and 2 columns
		fig, (ax1, ax2) = plt.subplots(1, 2)
		fig.set_size_inches(18, 7)

		# The 1st subplot is the silhouette plot
		# The silhouette coefficient can range from -1, 1 but in this example all
		# lie within [-0.1, 1]
		ax1.set_xlim([-0.1, 1])
		# The (n_clusters+1)*10 is for inserting blank space between silhouette
		# plots of individual clusters, to demarcate them clearly.
		ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

		#k_means = sklearn.cluster.DBSCAN(n_clusters=n_clusters,random_state=10)
		#k_means.fit(X_normalized)
		#cluster_labels = k_means.labels_
		knn_graph = kneighbors_graph(X_normalized, 5, include_self=False)
		model = AgglomerativeClustering(linkage='average',
										connectivity=knn_graph,
										n_clusters=n_clusters)

		model.fit(X_normalized)
		cluster_labels = model.labels_

		silhouette_avg = silhouette_score(X_normalized, cluster_labels)
		print("For n_clusters =", n_clusters,
			  "The average silhouette_score is :", silhouette_avg)

		# Compute the silhouette scores for each sample
		sample_silhouette_values = silhouette_samples(X_normalized, cluster_labels)

		#centroids = k_means.cluster_centers_

		# print('avg.deg \t avg.cc \t avg.com_size \t mod')
		# for row in centroids:
		# 	print([round(x,4) for x in row])


		for l in set(cluster_labels):
			print('Clusters', l)
			for i, ll in enumerate(cluster_labels):
				if ll == l:
					print('		',datasets[i])

		#
		#
		# #print(datasets)
		# y_lower = 10
		# for i in range(n_clusters):
		# 	# Aggregate the silhouette scores for samples belonging to
		# 	# cluster i, and sort them
		# 	ith_cluster_silhouette_values = \
		# 		sample_silhouette_values[cluster_labels == i]
		#
		# 	ith_cluster_silhouette_values.sort()
		#
		# 	size_cluster_i = ith_cluster_silhouette_values.shape[0]
		# 	y_upper = y_lower + size_cluster_i
		#
		# 	color = cm.spectral(float(i) / n_clusters)
		# 	ax1.fill_betweenx(np.arange(y_lower, y_upper),
		# 					  0, ith_cluster_silhouette_values,
		# 					  facecolor=color, edgecolor=color, alpha=0.7)
		#
		# 	# Label the silhouette plots with their cluster numbers at the middle
		# 	ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
		#
		# 	# Compute the new y_lower for next plot
		# 	y_lower = y_upper + 10  # 10 for the 0 samples
		#
		# ax1.set_title("The silhouette plot for the various clusters.")
		# ax1.set_xlabel("The silhouette coefficient values")
		# ax1.set_ylabel("Cluster label")
		#
		# # The vertical line for average silhouette score of all the values
		# ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
		#
		# ax1.set_yticks([])  # Clear the yaxis labels / ticks
		# ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
		#
		# # 2nd Plot showing the actual clusters formed
		# colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
		# ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
		# 			c=colors)
		#
		# # Labeling the clusters
		# centers = k_means.cluster_centers_
		# # Draw white circles at cluster centers
		# ax2.scatter(centers[:, 0], centers[:, 1],
		# 			marker='o', c="white", alpha=1, s=200)
		#
		# for i, c in enumerate(centers):
		# 	ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)
		#
		# ax2.set_title("The visualization of the clustered data.")
		# ax2.set_xlabel("Feature space for the 1st feature")
		# ax2.set_ylabel("Feature space for the 2nd feature")
		#
		# plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
		# 			  "with n_clusters = %d" % n_clusters),
		# 			 fontsize=14, fontweight='bold')
		#
		# plt.show()

	# for l in (set(labels)):
	# 	print('Group', l)
	# 	for i, ll in enumerate(labels):
	# 		if l==ll:
	# 			print(datasets[i])
	# 	print('-'*10)

		#print(datasets[i], '\t', l)
	#
	# print(labels)
	# estimators = {'k_means_iris_3': KMeans(n_clusters=3),
	# 			  'k_means_iris_8': KMeans(n_clusters=8),
	# 			  'k_means_iris_bad_init': KMeans(n_clusters=3, n_init=1,
	# 											  init='random')}
	#
	# fignum = 1
	# for name, est in estimators.items():
	# 	fig = plt.figure(fignum, figsize=(4, 3))
	# 	plt.clf()
	# 	ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
	#
	# 	plt.cla()
	# 	est.fit(X)
	# 	labels = est.labels_
	#
	# 	ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))
	#
	# 	ax.w_xaxis.set_ticklabels([])
	# 	ax.w_yaxis.set_ticklabels([])
	# 	ax.w_zaxis.set_ticklabels([])
	# 	ax.set_xlabel('Petal width')
	# 	ax.set_ylabel('Sepal length')
	# 	ax.set_zlabel('Petal length')
	# 	fignum = fignum + 1