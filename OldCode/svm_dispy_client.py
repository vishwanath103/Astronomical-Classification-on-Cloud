
def svm_func(data, classes):
    import socket
    from sklearn import svm
    host = socket.gethostname()
    rbf_svm = svm.SVC(C=10, kernel='rbf')
    rbf_svm.fit(data, classes)
    return (host, rbf_svm.support_)

if __name__ == '__main__':
	import dispy, pandas as pd
	from sklearn import svm
	
	data = pd.read_csv('training_data.csv', header=0)
	train_data = data[0:250000]
	d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 
	'u-r' : pd.Series(train_data['u']-train_data['r']),
	'r-i' : pd.Series(train_data['r']-train_data['i']),
	'i-z' : pd.Series(train_data['i']-train_data['z'])}
	df = pd.DataFrame(d)
	#df = scaler.fit_transform(df)
	#df = pd.DataFrame(df)
	classes = []
	classes = ['QSO' if train_data['class'][i]=='QSO' else 'NQSO' for i in range(len(train_data['class']))]
	#for i in range(len(train_data['class'])):
	#	if(train_data['class'][i]=='QSO'):
	#		classes.append('QSO')
	#	else:
	#		classes.append('NQSO')
	#	print i
	test_data = data[400000:]
	d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 
	'u-r' : pd.Series(test_data['u']-test_data['r']), 
	'r-i' : pd.Series(test_data['r']-test_data['i']),
	'i-z' : pd.Series(test_data['i']-test_data['z'])}
	df2 = pd.DataFrame(d)
	#df2 = scaler.fit_transform(df2)
	#df2 = pd.DataFrame(df2)
	classes2 = pd.Series(test_data['class'])

	cluster = dispy.JobCluster(svm_func)
	jobs = []
	for i in range(4):
		job = cluster.submit(df[(i*62500):((i+1)*62500)], classes[(i*62500):((i+1)*62500)])
		job.id = i
		jobs.append(job)

	l = []
	for job in jobs:
		host, result = job() # waits for job to finish and returns results
		for i in result:
			l.append(i+(job.id*62500))
		print('%s executed job %s at %s' % (host, job.id, job.start_time))
	cluster.stats()
	new_df = df.iloc[l, 0:4]
	new_classes = []
	for i in l:
		new_classes.append(classes[i])
	
	#print len(l)
	#jobs = []
	#for i in range(len(l)/40000 + 1):
	#	if(i == len(l)/40000):
	#		job = cluster.submit(new_df[(i*40000):], new_classes[(i*40000):])
	#	else:
	#		job = cluster.submit(new_df[(i*40000):((i+1)*40000)], new_classes[(i*40000):((i+1)*40000)])
	#	job.id = i
	#	jobs.append(job)
		
	#l = []
	#for job in jobs:
	#	host, result = job() # waits for job to finish and returns results
	#	for i in result:
	#		l.append(i+(job.id*40000))
	#	print('%s executed job %s at %s' % (host, job.id, job.start_time))
	#cluster.stats()
	#new_new_df = new_df.iloc[l, 0:4]
	#new_new_classes = []
	
	print len(l)
	
	#for i in l:
	#	new_new_classes.append(new_classes[i])
	
	rbf_svm = svm.SVC(kernel='rbf')	
	rbf_svm.fit(new_df, new_classes)
	print 'here'
	predicted_class = rbf_svm.predict(df2)
	print 'here'
	
	tp,tn,fp,fn = 0,0,0,0
	for i in range(0, len(classes2)):
		if(predicted_class[i] == 'NQSO' and classes2[400000+i]!= 'QSO'):
			tn += 1
		elif(predicted_class[i] == 'QSO' and classes2[400000+i]== 'QSO'):
			tp += 1
		elif(predicted_class[i] == 'NQSO' and classes2[400000+i]== 'QSO'):
			fn += 1
		elif(predicted_class[i] == 'QSO' and classes2[400000+i]!= 'QSO'):
			fp += 1
	print ''
	print 'SVM results:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn), 'accuracy='+str(((tp+tn)*100.0)/(tp+fp+fn+tn))
