import multiprocessing as mp

def job(n):
    global param
    return n+param.value

def init(arg):
    global param #global parameter that is shared between processes
    param = arg

if __name__=='__main__':
    w = mp.Value('d', 1.0)
    pool = mp.Pool(initializer=init, initargs=(w,))
    res = pool.map(job, range(10))
    print(res)