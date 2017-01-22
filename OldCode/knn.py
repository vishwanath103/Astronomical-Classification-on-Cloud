import pandas as pd
from sklearn.neighbors import NearestNeighbors
import numpy as np

data = pd.read_csv('MyTable_anisha.csv', header=0)
train_data = data[0:3200000]
d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 
	 'u-r' : pd.Series(train_data['u']-train_data['r']),
	 'r-i' : pd.Series(train_data['r']-train_data['i']),
	 'i-z' : pd.Series(train_data['i']-train_data['z'])}
df = pd.DataFrame(d)
print 'Training Data:'
print df[:3]
print 'Shape='+str(df.shape)
classes = pd.Series(train_data['class'])

test_data = data[3500000:3500100]
d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 
	 'u-r' : pd.Series(test_data['u']-test_data['r']),
	 'r-i' : pd.Series(test_data['r']-test_data['i']),
	 'i-z' : pd.Series(test_data['i']-test_data['z'])}
df2 = pd.DataFrame(d)
print ''
print 'Testing Data:'
print df2[:3]
print 'Shape='+str(df2.shape)
classes2 = pd.Series(test_data['class'])

nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(df)
distances, indices = nbrs.kneighbors(df2)
predicted_class = []

for i in range(indices.size/7):
	quasar_count = 0
	non_quasar_count = 0
	for j in range(indices[i].size):
		if(classes[indices[i,j]] == 'QSO'):
			quasar_count += (1/(distances[i,j]*distances[i,j]))
		else:
			non_quasar_count += (1/(distances[i,j]*distances[i,j]))
	if(quasar_count > non_quasar_count):
			predicted_class.append('QSO')
	else:
			predicted_class.append('NQSO')

tp,tn,fp,fn = 0,0,0,0
for i in range(0, len(classes2)):
	if(predicted_class[i] == 'NQSO' and classes2[3500000+i]!= 'QSO'):
		tn += 1
	elif(predicted_class[i] == 'QSO' and classes2[3500000+i]== 'QSO'):
		tp += 1
	elif(predicted_class[i] == 'NQSO' and classes2[3500000+i]== 'QSO'):
		fn += 1
	else:
		fp += 1
print 
print '7NN results:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn), 'accuracy='+str(((tp+tn)*100.0)/(tp+fp+fn+tn))
