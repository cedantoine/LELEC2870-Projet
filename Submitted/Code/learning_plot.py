# *-* coding: utf-8 *-*

import math as mt
import numpy as np
import pandas as pd

import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
nfig = 1

lin_file_list = {"method":"linear", "featuring":["corr", "pca", "mutual_info"]}
knn_file_list = {"method":"knn", "featuring":["pca", "mutual_info"]}
mlp_file_list = {"method":"mlp", "featuring":["pca", "mutual_info"]}
pls_file_list = {"method":"pls", "featuring":["pca", "mutual_info"]}

scmin = 45
scmax = 50
level_cf = MaxNLocator(nbins=20).tick_values(scmin,scmax)
level_tc = np.arange(scmin,scmax)

def filename(method, featuring):
	return "Scores/" + method + "_" + featuring + ".txt"

def figname(method, featuring=None, learning=None):
	if featuring==None:
		return 'Figs/' + method + '.pdf'
	elif learning==None:
		return 'Figs/' + method + '_' + featuring + '.pdf'
	else:
		return 'Figs/' + method + '_' + featuring + '_' + learning + '.pdf'

def len_of_var(var):
	res = []
	for x in var:
		if x not in res:
			res.append(x)
	return len(res)

try:
	##
	fig = plt.figure(nfig)
	nfig+= 1
	for featuring in lin_file_list["featuring"]:
		method = lin_file_list["method"]
		with open(filename(method, featuring), 'r') as file:
			varbl = file.readline()[:-1].split(", ")
			lines = file.readlines()

			datas = np.zeros((len(lines),len(varbl)))

			for (i,l) in enumerate(lines):
				datas[i] = [float(x) for x in l[:-1].split(", ")]

			score = datas[:,0].squeeze()*100
			feats = datas[:,1].squeeze()
			plt.plot(feats, score, '-o', label=featuring)

	plt.xlabel("amount of features")
	plt.ylabel("score [%]")
	plt.legend()
	plt.tight_layout()

	fig.savefig(figname(method))

	##
	for featuring in knn_file_list["featuring"]:
		method = knn_file_list["method"]

		with open(filename(method, featuring), 'r') as file:
			varbl = file.readline()[:-1].split(", ")
			lines = file.readlines()

			datas = np.zeros((len(lines),len(varbl)))
			
			for (i,l) in enumerate(lines):
				datas[i] = [float(x) for x in l[:-1].split(", ")]

			var_l = [len_of_var(datas[:,2]), len_of_var(datas[:,1]), 3]
			datas = datas.reshape(var_l)

			score = datas[:,:,0]*100
			feats = datas[:,:,1]
			neigh = datas[:,:,2]

		fig = plt.figure(nfig)
		cf  = plt.contourf(feats,neigh, score, levels=level_cf)
		fig.colorbar(cf, ticks=level_tc, format='%d') #, ax=ax)

		plt.xlabel("amount of features")
		plt.ylabel("amount of neighbours")
		plt.tight_layout()
		nfig+= 1

		fig.savefig(figname(method, featuring))

	##
	for featuring in mlp_file_list["featuring"]:
		method = mlp_file_list["method"]

		with open(filename(method, featuring), 'r') as file:
			varbl = file.readline()[:-1].split(", ")
			score = []
			feats = []
			layer = []
			learn = []
			for l in file.readlines():
				datas = l[:-1].split(', ')
				score.append(float(datas[0]))
				feats.append(int(datas[1]))
				try:
					layer.append(int(datas[2]))
				except:
					x = datas[2].split(',')
					layer.append((int(x[0][1:]),
								  int(x[1][:-1])))
				learn.append(datas[3])

			nb_feats = len_of_var(feats)
			nb_layer = len_of_var(layer)
			nb_learn = len_of_var(learn)

			score = np.array(score).reshape(nb_layer,nb_feats,nb_learn)*100
			feats = np.array(feats).reshape(nb_layer,nb_feats,nb_learn)
			layer = np.array(layer).reshape(nb_layer,nb_feats,nb_learn)
			layer_lab = {x:i for i,x in enumerate(layer[:,0,0])}
			for i,layer1 in enumerate(layer):
				for j,layer2 in enumerate(layer1):
					for k,layer3 in enumerate(layer2):
						layer[i,j,k] = layer_lab[layer3]
			learn = np.array(learn).reshape(nb_layer,nb_feats,nb_learn)

			for i in range(nb_learn):
				fig = plt.figure(nfig)
				plt.contourf(feats[:,:,i],layer[:,:,i], score[:,:,i], levels=level_cf)
				fig.colorbar(cf, ticks=level_tc, format='%d') #, ax=ax)
				plt.xlabel("amount of features")
				plt.ylabel("layers repartition")
				plt.yticks(list(layer_lab.values()), list(layer_lab.keys()))
				plt.tight_layout()
				nfig+= 1

				fig.savefig(figname(method, featuring, learn[0,0,i]))

	##
	for featuring in pls_file_list["featuring"]:
		method = pls_file_list["method"]

		with open(filename(method, featuring), 'r') as file:
			varbl = file.readline()[:-1].split(", ")
			lines = file.readlines()

			datas = np.zeros((len(lines),len(varbl)))

			for (i,l) in enumerate(lines):
				datas[i] = [float(x) for x in l[:-1].split(", ")]

			var_l = [len_of_var(datas[:,2]), len_of_var(datas[:,1]), 3]
			datas = datas.reshape(var_l)

			score = datas[:,:,0]*100
			feats = datas[:,:,1]
			compt = datas[:,:,2]

		
		fig = plt.figure(nfig)
		plt.contourf(feats,compt, score, levels=level_cf)
		fig.colorbar(cf, ticks=level_tc, format='%d') #, ax=ax)
		plt.xlabel("amount of features")
		plt.ylabel("amount of components")
		plt.tight_layout()
		nfig+= 1

		fig.savefig(figname(method, featuring))
except FileNotFoundError as err:
	print('We could not find some ".txt" files')
	print("Please read carefully the README file and completly run the jupyter notebook")
	quit()

if nfig>1: plt.show()
