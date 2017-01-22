
def knn(jobid):
    import socket
    import dispy, pandas as pd
    from sklearn.neighbors import NearestNeighbors
    data2 = pd.read_csv('/home/anisha/College/Distributed Computing/Project/MyTable_anisha.csv', header=0)
    train_data = data2[0:3000000]
    d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 'u-r' : pd.Series(train_data['u']-train_data['r']), 'r-i' : pd.Series(train_data['r']-train_data['i']),'i-z' : pd.Series(train_data['i']-train_data['z'])}
    data = pd.DataFrame(d)
    data = data[(jobid*100000):((jobid+1)*100000)]
    classes = pd.Series(train_data['class'])
    test_data = data2[3500000:]
    d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 'u-r' : pd.Series(test_data['u']-test_data['r']),'r-i' : pd.Series(test_data['r']-test_data['i']),'i-z' : pd.Series(test_data['i']-test_data['z'])}
    testdata = pd.DataFrame(d)
    nbrs = NearestNeighbors(n_neighbors=7, algorithm='auto').fit(data)
    host = socket.gethostname()
    distances, indices = nbrs.kneighbors(testdata)
    f = open('/home/anisha/College/Distributed Computing/Project/result'+str(jobid)+'.txt', 'w')
    for i in range(len(indices)):
		temp = []
		for j in range(len(indices[i])):
			f.write(str(distances[i][j]) + "," + str(indices[i][j]+jobid*100000) + " ")
		f.write("\n")
    host = socket.gethostname()
    f.close()
    #dispy_send_file('result'+str(jobid)+'.txt')
    return (host)

if __name__ == '__main__':
	import dispy, pandas as pd
	from sklearn.neighbors import NearestNeighbors
	
	data = pd.read_csv('MyTable_anisha.csv', header=0)
	train_data = data[0:3000000]
	d = {'g-r' : pd.Series(train_data['g']-train_data['r']), 'u-r' : pd.Series(train_data['u']-train_data['r']), 'r-i' : pd.Series(train_data['r']-train_data['i']),'i-z' : pd.Series(train_data['i']-train_data['z'])}
	df = pd.DataFrame(d)
	classes = pd.Series(train_data['class'])
	test_data = data[3500000:]
	d = {'g-r' : pd.Series(test_data['g']-test_data['r']), 'u-r' : pd.Series(test_data['u']-test_data['r']),'r-i' : pd.Series(test_data['r']-test_data['i']),'i-z' : pd.Series(test_data['i']-test_data['z'])}
	df2 = pd.DataFrame(d)

	cluster = dispy.JobCluster(knn)
	jobs = []
	for i in range(30):
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
		for j in range(7):
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
		if(predicted_class[i] == 'NQSO' and classes2[3500000+i]!= 'QSO'):
			tn += 1
		elif(predicted_class[i] == 'QSO' and classes2[3500000+i]== 'QSO'):
			tp += 1
		elif(predicted_class[i] == 'NQSO' and classes2[3500000+i]== 'QSO'):
			fn += 1
		else:
			fp += 1
	print ''
	print '7NN results:', 'tp='+str(tp), 'fp='+str(fp), 'tn='+str(tn), 'fn='+str(fn)

