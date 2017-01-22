
def svm_func(data, classes):
    import socket
    from sklearn import svm
    host = socket.gethostname()
    rbf_svm = svm.SVC(C=10, kernel='rbf', gamma=1)
    rbf_svm.fit(data, classes)
    return (host, rbf_svm.support_)

def knn(jobid):
    import socket
    import dispy, pandas as pd
    from sklearn.neighbors import NearestNeighbors
    data2 = pd.read_csv('/home/anisha/College/Distributed Computing/Project/MyTable_anisha', header=0)
    train_data = data2[:3500000]
    d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 'u-r' : pd.Series(train_data['u']-train_data['r']), 'r-i' : pd.Series(train_data['r']-train_data['i']),'i-z' : pd.Series(train_data['i']-train_data['z'])}
    data = pd.DataFrame(d)
    data = data[500000+(jobid*300000):500000+((jobid+1)*300000)]
    classes = pd.Series(train_data['class'])
    test_data = data2[400000:500000]
    d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 'u-r' : pd.Series(test_data['u']-test_data['r']),'r-i' : pd.Series(test_data['r']-test_data['i']),'i-z' : pd.Series(test_data['i']-test_data['z'])}
    testdata = pd.DataFrame(d)
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(data)
    host = socket.gethostname()
    distances, indices = nbrs.kneighbors(testdata)
    f = open('/home/anisha/College/Distributed Computing/Project/result'+str(jobid)+'.txt', 'w')
    for i in range(len(indices)):
		temp = []
		for j in range(len(indices[i])):
			f.write(str(distances[i][j]) + "," + str(500000+indices[i][j]+jobid*300000) + " ")
		f.write("\n")
    host = socket.gethostname()
    f.close()
    #dispy_send_file('result'+str(jobid)+'.txt')
    return (host)

if __name__ == '__main__':
	import dispy, pandas as pd
	from sklearn import svm
	from sklearn.neighbors import NearestNeighbors
	
	data = pd.read_csv('MyTable_anisha.csv', header=0)
	test_data = data[400000:500000]
	d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 
	'u-r' : pd.Series(test_data['u']-test_data['r']), 
	'r-i' : pd.Series(test_data['r']-test_data['i']),
	'i-z' : pd.Series(test_data['i']-test_data['z'])}
	df2 = pd.DataFrame(d)
	classes2 = pd.Series(test_data['class'])
	predicted_classes = []
	train_data = data[500000:3500000]
	d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 'u-r' : pd.Series(train_data['u']-train_data['r']), 'r-i' : pd.Series(train_data['r']-train_data['i']),'i-z' : pd.Series(train_data['i']-train_data['z'])}
	df = pd.DataFrame(d)
	classes = pd.Series(train_data['class'])
	
	cluster = dispy.JobCluster(knn)
	jobs = []
	for i in range(10):
		job = cluster.submit(i)
		job.id = i
		jobs.append(job)

	l = []
	for job in jobs:
		host = job() # waits for job to finish and returns results
		f = open('result'+str(job.id)+'.txt', 'r')
		lines = f.readlines()
		indices = []
		count = 0
		for l1 in lines[:len(lines)]:
			points = l1.split()
			indices.append([])
			for p in points:
				indices[count].append([float(p.split(',')[0]),int(p.split(',')[1])])
			count = count + 1
		l.append(indices)
		f.close()
		print('%s executed job %s at %s' % (host, job.id, job.start_time))
	cluster.stats()
	#classes2 = pd.Series(test_data['class'])
	predicted_class = []
	for r in range(len(test_data)):
		quasar_count = 0
		non_quasar_count = 0
		result = []
		for i in range(len(l)):
			for j in l[i][r]:
				result.append(j)
		result.sort()
		for j in range(7):
			if(classes[result[j][1]] == 'QSO'):
				quasar_count += 1 #(1/(result[j][0]*result[j][0]))
			else:
				non_quasar_count += 1 #(1/(result[j][0]*result[j][0]))
		if(quasar_count > non_quasar_count):
			predicted_class.append('QSO')
		else:
			predicted_class.append('NQSO')	
	predicted_classes.append(predicted_class)
	
	for n in range(1,7):
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
		for j in range(7):
			if(predicted_classes[j][i] == 'NQSO'):
				res_class = 'NQSO'
				break
		if(res_class == 'NQSO' and classes2[400000+i]!= 'QSO'):
			tn += 1
		elif(res_class == 'QSO' and classes2[400000+i]== 'QSO'):
			tp += 1
		elif(res_class == 'NQSO' and classes2[400000+i]== 'QSO'):
			fn += 1
		elif(res_class == 'QSO' and classes2[400000+i]!= 'QSO'):
			fp += 1
	print ''
	print 'SVM+KNN results 1:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn), 'accuracy='+str(((tp+tn)*100.0)/(tp+fp+fn+tn))
	
	tp,tn,fp,fn = 0,0,0,0
	for i in range(0, len(predicted_classes[0])):
		res_class_nqso = 0
		res_class_qso = 0
		for j in range(7):
			if(predicted_classes[j][i] == 'NQSO'):
				res_class_nqso += 1
			else:
				res_class_qso += 1
		if(res_class_nqso>=res_class_qso and classes2[400000+i]!= 'QSO'):
			tn += 1
		elif(res_class_nqso<res_class_qso and classes2[400000+i]== 'QSO'):
			tp += 1
		elif(res_class_nqso>=res_class_qso and classes2[400000+i]== 'QSO'):
			fn += 1
		elif(res_class_nqso<res_class_qso and classes2[400000+i]!= 'QSO'):
			fp += 1
	print ''
	print 'SVM+KNN results 2:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn), 'accuracy='+str(((tp+tn)*100.0)/(tp+fp+fn+tn))
	
	#data = pd.read_csv('testing_data.csv', header=0)
	#test_data = data[0:500]
	#d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 'u-r' : pd.Series(test_data['u']-test_data['r']),'r-i' : pd.Series(test_data['r']-test_data['i']),'i-z' : pd.Series(test_data['i']-test_data['z'])}
	#df = pd.DataFrame(d)
	#test_data.loc[:,'predicted_class'] = pd.Series(rbf_svm.predict(df), index=test_data.index)
	#print test_data[0:10]
	



