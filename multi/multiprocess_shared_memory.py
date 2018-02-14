import multiprocessing as mp

def job(n, w, l):
    l.acquire()
    w.value += n
    print(w.value)
    l.release()

if __name__=='__main__':
    w = mp.Value('d', 0.0)
    l = mp.Lock()
    p1 = mp.Process(target=job, args=(1, w, l))
    p2 = mp.Process(target=job, args=(2, w, l))
    p1.start()
    p2.start()
    p1.join()
    p2.join()