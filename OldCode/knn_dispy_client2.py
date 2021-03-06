
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
    f = open('result'+str(jobid)+'.txt', 'w')
    for i in range(len(indices)):
		temp = []
		for j in range(len(indices[i])):
			f.write(str(distances[i][j]) + "," + str(indices[i][j]+jobid*20000) + " ")
		f.write("\n")
    host = socket.gethostname()
    f.close()
    dispy_send_file('result'+str(jobid)+'.txt')
    return (host)

if __name__ == '__main__':
	# executed on client only; variables created below, including modules imported,
	# are not available in job computations
	import dispy, pandas as pd
	from sklearn.neighbors import NearestNeighbors
	cluster = dispy.JobCluster(knn)
	#import dispy.httpd
	#http_server = dispy.httpd.DispyHTTPServer(cluster)
	data = pd.read_csv('training_data.csv', header=0)
	train_data = data[0:400000]
	d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 'u-r' : pd.Series(train_data['u']-train_data['r']), 'r-i' : pd.Series(train_data['r']-train_data['i']),'i-z' : pd.Series(train_data['i']-train_data['z'])}
	df = pd.DataFrame(d)
	classes = pd.Series(train_data['class'])
	test_data = data[400000:]
	d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 'u-r' : pd.Series(test_data['u']-test_data['r']),'r-i' : pd.Series(test_data['r']-test_data['i']),'i-z' : pd.Series(test_data['i']-test_data['z'])}
	df2 = pd.DataFrame(d)
	jobs = []
	for i in range(20):
		job = cluster.submit(df[(i*20000):((i+1)*20000)])
		job.id = i # associate an ID to identify jobs (if needed later)
		jobs.append(job)
	#cluster.wait() # waits until all jobs finish
	l = []
	for job in jobs:
		host, nbr = job() # waits for job to finish and returns results
		l.append(nbr);
		print('%s executed job %s at %s' % (host, job.id, job.start_time))
		# other fields of 'job' that may be useful:
		# job.stdout, job.stderr, job.exception, job.ip_addr, job.end_time
	#cluster.wait()
	cluster.stats()

	cluster2 = dispy.JobCluster(testing)
	jobs = []
	for i in range(20):
		job = cluster2.submit(df2, l[i], i)
		job.id = i # associate an ID to identify jobs (if needed later)
		jobs.append(job)
	#cluster2.wait() # waits until all jobs finish
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
