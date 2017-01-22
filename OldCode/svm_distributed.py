
def knn(data):
    import socket
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=3, algorithm='auto').fit(data)
    host = socket.gethostname()
    return (host, nbrs)
    
def testing(data, nbrs, jobid):
    import socket
    from sklearn.neighbors import NearestNeighbors
    distances, indices = nbrs.kneighbors(data)
    result = []
    for i in range(len(indices)):
		temp = []
		for j in range(len(indices[i])):
			temp.append([distances[i][j],indices[i][j]+jobid*100000])
		result.append(temp)
    host = socket.gethostname()
    return (host, result)

if __name__ == '__main__':	import dispy, pandas as pd
	from sklearn.neighbors import NearestNeighbors
	cluster = dispy.JobCluster(knn)
	data = pd.read_csv('training_data.csv', header=1)
	train_data = data[0:400000]
	d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 'u-r' : pd.Series(train_data['u']-train_data['r']), 'r-i' : pd.Series(train_data['r']-train_data['i']),'i-z' : pd.Series(train_data['i']-train_data['z'])}
	df = pd.DataFrame(d)
	classes = pd.Series(train_data['class'])
	test_data = data[400000:]
	d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 'u-r' : pd.Series(test_data['u']-test_data['r']),'r-i' : pd.Series(test_data['r']-test_data['i']),'i-z' : pd.Series(test_data['i']-test_data['z'])}
	df2 = pd.DataFrame(d)
	jobs = []
	for i in range(4):
		job = cluster.submit(df[(i*100000):((i+1)*100000)])
		job.id = i # associate an ID to identify jobs (if needed later)
		jobs.append(job)
	l = []
	for job in jobs:
		host, nbr = job() # waits for job to finish and returns results
		l.append(nbr);
		print('%s executed job %s at %s' % (host, job.id, job.start_time))
	cluster.stats()

	cluster2 = dispy.JobCluster(testing, nodes=['192.16.112.7','192.16.112.5'])
	jobs = []
	for i in range(4):
		job = cluster2.submit(df2, l[i], i)
		job.id = i # associate an ID to identify jobs (if needed later)
		jobs.append(job)
	l = []
	for job in jobs:
		host, indices = job() # waits for job to finish and returns results
		l.append(indices)
		print('%s executed job %s at %s' % (host, job.id, job.start_time))
	cluster2.stats()
	
	classes2 = pd.Series(test_data['class'])
	predicted_class = []
	for r in range(len(test_data)):
		quasar_count = 0
		non_quasar_count = 0
		result = []
		for i in range(len(l)):
			for j in l[i][r]:
				result.append(j)
		result.sort()
		for j in range(3):
			if(classes[result[j][1]] == 'QSO'):
				quasar_count += 1
			else:
				non_quasar_count += 1
		if(quasar_count > non_quasar_count):
			predicted_class.append('QSO')
		else:
			predicted_class.append('NQSO')
	tp,tn,fp,fn = 0,0,0,0
	for i in range(0, len(classes2)):
		if(predicted_class[i] == 'NQSO' and classes2[400000+i]!= 'QSO'):
			tn += 1
		elif(predicted_class[i] == 'QSO' and classes2[400000+i]== 'QSO'):
			tp += 1
		elif(predicted_class[i] == 'NQSO' and classes2[400000+i]== 'QSO'):
			fn += 1
		else:
			fp += 1
	print ''
	print '3NN results:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn)
