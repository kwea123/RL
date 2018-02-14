import multiprocessing as mp

def job(n):
    return n*n

if __name__=='__main__':
    pool = mp.Pool()
    res = pool.map(job, range(1000000))
    #print(res)