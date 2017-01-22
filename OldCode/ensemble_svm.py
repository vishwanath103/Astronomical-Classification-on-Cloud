
def svm_func(data, classes):
    import socket
    from sklearn import svm
    host = socket.gethostname()
    rbf_svm = svm.SVC(C=5, kernel='rbf', gamma=0)
    rbf_svm.fit(data, classes)
    return (host, rbf_svm.support_)

if __name__ == '__main__':
	import dispy, pandas as pd
	from sklearn import svm
	
	data = pd.read_csv('MyTable_anisha.csv', header=0)
	test_data = data[500000:510000]
	d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 
	'u-r' : pd.Series(test_data['u']-test_data['r']), 
	'r-i' : pd.Series(test_data['r']-test_data['i']),
	'i-z' : pd.Series(test_data['i']-test_data['z'])}
	df2 = pd.DataFrame(d)
	classes2 = pd.Series(test_data['class'])
	predicted_classes = []
	
	for n in range(1,2):
		train_data = data[n*500000:(n+1)*500000]
		d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 
		'u-r' : pd.Series(train_data['u']-train_data['r']),
		'r-i' : pd.Series(train_data['r']-train_data['i']),
		'i-z' : pd.Series(train_data['i']-train_data['z'])}
		df = pd.DataFrame(d)
		classes = []
		classes = ['QSO' if train_data['class'][j+n*500000]=='QSO' else 'NQSO' for j in range(len(train_data['class']))]

		cluster = dispy.JobCluster(svm_func)
		jobs = []
		for i in range(10):
			job = cluster.submit(df[(i*50000):((i+1)*50000)], classes[(i*50000):((i+1)*50000)])
			job.id = i
			jobs.append(job)

		l = []
		for job in jobs:
			host, result = job() # waits for job to finish and returns results
			for i in result:
				l.append(i+(job.id*50000))
			print('%s executed job %s at %s' % (host, job.id, job.start_time))
		cluster.stats()
		new_df = df.iloc[l, 0:4]
		new_classes = []
		for i in l:
			new_classes.append(classes[i])
		
		print len(l)
		
		rbf_svm = svm.SVC(kernel='rbf')	
		rbf_svm.fit(new_df, new_classes)
		print 'here'
		print len(rbf_svm.support_)
		predicted_class = rbf_svm.predict(df2)
		print 'here'
		predicted_classes.append(predicted_class)
		
		
	tp,tn,fp,fn = 0,0,0,0
	for i in range(0, len(predicted_classes[0])):
		res_class = 'QSO'
		for j in range(1):
			if(predicted_classes[j][i] == 'NQSO'):
				res_class = 'NQSO'
				break
		if(res_class == 'NQSO' and classes2[500000+i]!= 'QSO'):
			tn += 1
		elif(res_class == 'QSO' and classes2[500000+i]== 'QSO'):
			tp += 1
		elif(res_class == 'NQSO' and classes2[500000+i]== 'QSO'):
			fn += 1
		elif(res_class == 'QSO' and classes2[500000+i]!= 'QSO'):
			fp += 1
		'''
		res_class_nqso = 0
		res_class_qso = 0
		for j in range(1):
			if(predicted_classes[j][i] == 'NQSO'):
				res_class_nqso += 1
			else:
				res_class_qso += 1
		if(res_class_nqso>=res_class_qso and classes2[3500000+i]!= 'QSO'):
			tn += 1
		elif(res_class_nqso<res_class_qso and classes2[3500000+i]== 'QSO'):
			tp += 1
		elif(res_class_nqso>=res_class_qso and classes2[3500000+i]== 'QSO'):
			fn += 1
		elif(res_class_nqso<res_class_qso and classes2[3500000+i]!= 'QSO'):
			fp += 1
		'''
	print ''
	print 'SVM results:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn), 'accuracy='+str(((tp+tn)*100.0)/(tp+fp+fn+tn))
	
	#data = pd.read_csv('testing_data.csv', header=0)
	#test_data = data[0:500]
	#d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 'u-r' : pd.Series(test_data['u']-test_data['r']),'r-i' : pd.Series(test_data['r']-test_data['i']),'i-z' : pd.Series(test_data['i']-test_data['z'])}
	#df = pd.DataFrame(d)
	#test_data.loc[:,'predicted_class'] = pd.Series(rbf_svm.predict(df), index=test_data.index)
	#print test_data[0:10]
	


