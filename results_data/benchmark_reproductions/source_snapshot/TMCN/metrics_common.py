import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.cluster import contingency_matrix
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, fowlkes_mallows_score, homogeneity_score, completeness_score, v_measure_score
def evaluate(y,p):
 y=np.asarray(y).ravel();p=np.asarray(p).ravel()
 assert len(y)==len(p) and len(y)>0
 c=contingency_matrix(y,p);r,k=linear_sum_assignment(-c)
 return {'ACC':float(c[r,k].sum()/len(y)),'NMI':normalized_mutual_info_score(y,p,average_method='arithmetic'),'ARI':adjusted_rand_score(y,p),'PUR':float(c.max(axis=0).sum()/len(y)),'FMI':fowlkes_mallows_score(y,p),'HI':homogeneity_score(y,p),'CI':completeness_score(y,p),'VM':v_measure_score(y,p)}
