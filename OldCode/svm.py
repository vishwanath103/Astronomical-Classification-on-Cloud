
import pandas as pd
from sklearn import svm
import numpy as np

data = pd.read_csv('training_data.csv', header=0)
train_data = data[0:40000]
d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 
	 'u-r' : pd.Series(train_data['u']-train_data['r']),
	 'r-i' : pd.Series(train_data['r']-train_data['i']),
	 'i-z' : pd.Series(train_data['i']-train_data['z'])}
df = pd.DataFrame(d)
print 'Training Data:'
print df[:3]
print 'Shape='+str(df.shape)
classes = []
for i in range(len(train_data['class'][:40000])):
	if(train_data['class'][i]=='QSO'):
		classes.append('QSO')
	else:
		classes.append('NQSO')
data2 = pd.read_csv('testing_data.csv', header=0)
test_data = data2[0:500000]
d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 
	 'u-r' : pd.Series(test_data['u']-test_data['r']),
	 'r-i' : pd.Series(test_data['r']-test_data['i']),
	 'i-z' : pd.Series(test_data['i']-test_data['z'])}
df2 = pd.DataFrame(d)
rbf_svm = svm.SVC(kernel='rbf')
rbf_svm.fit(df, classes)
predicted_class = rbf_svm.predict(df2)
print list(predicted_class).count('QSO')
print len(data2)
print len(predicted_class)
test_data["Predicted Class"] = predicted_class
test_data.to_csv("final_output.csv")
