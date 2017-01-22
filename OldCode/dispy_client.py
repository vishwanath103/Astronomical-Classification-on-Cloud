
def compute(n):
    import time, socket
    time.sleep(n)
    host = socket.gethostname()
    return (host, n)

if __name__ == '__main__':
    # executed on client only; variables created below, including modules imported,
    # are not available in job computations
    import dispy, random
    cluster = dispy.JobCluster(compute)
    #import dispy.httpd
    #http_server = dispy.httpd.DispyHTTPServer(cluster)
    jobs = []
    for i in range(20):
        job = cluster.submit(random.randint(5,20))
        job.id = i # associate an ID to identify jobs (if needed later)
        jobs.append(job)
    #cluster.wait() # waits until all jobs finish
    for job in jobs:
        host, n = job() # waits for job to finish and returns results
        print('%s executed job %s at %s with %s' % (host, job.id, job.start_time, n))
        # other fields of 'job' that may be useful:
        # job.stdout, job.stderr, job.exception, job.ip_addr, job.end_time
    cluster.stats()
