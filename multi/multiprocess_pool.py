import multiprocessing as mp

def job(q, n):
    res=0
    for i in range(n):
        res+=i
    q.put(res)    #queue

if __name__=='__main__':
    q = mp.Queue()
    p1 = mp.Process(target=job,args=(q, 10))
    p2 = mp.Process(target=job,args=(q, 100))
    p1.start()
    p2.start()
    p1.join()
    p2.join()
    res1 = q.get()
    res2 = q.get()
    print(res1)
    print(res2)