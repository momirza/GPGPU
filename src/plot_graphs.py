from subprocess import Popen, PIPE
from matplotlib import pyplot
import csv
import numpy as np
from scipy import interpolate
import sys




RUNS = 10

def dump_csvs(times):
    """ takes in data as list and generates csv """
    writer = csv.writer(open('mat.csv', 'w'))
    for n, time in enumerate(times):
        writer.writerow([squares[n]] + time)

def call_exec(file):
    proc_list = []
    get_times = lambda x: map(float, x.split())
    for _ in xrange(RUNS):
        proc_list.append(map(get_times, (Popen([('./%s' %file)] , stdout=PIPE).stdout.readlines())))
    averaged = [[map(np.mean, zip(*[x[y] for x in proc_list]))] for y in range(len(proc_list[0]))]
    averaged = [x[0] for x in averaged]
    return [[[int(x[1]), x[2]] for x in averaged if int(x[0])==y] for y in range(3)]
    # return [[[int(x[1]), np.log10(0.000001) if not x[2] else np.log10(x[2])] for x in averaged if int(x[0])==y] for y in range(3)]

if __name__ == "__main__":
    DATA_EXISTS = False if sys.argv[-1] == 'rerun' else True

    if DATA_EXISTS:
        reader = csv.reader(open('data.csv', 'r'))
        times = []
        for row in reader:
            times.append(map(float, row))
        squares, ser, tbb, opt, eli = zip(*times)  
    else:
        F0, F1, F2 = call_exec("dumb")
        # dump_csvs(times)
    nF0, F0 = zip(*F0)
    nF1, F1 = zip(*F1)
    nF2, F2 = zip(*F2)
    # nF0, F0 = zip(*F0)
    # nF0, F0 = zip(*F0)
    # nF0, F0 = zip(*F0)

    
    tck_F0 = interpolate.splrep(nF0,F0,s=0)
    tck_F1 = interpolate.splrep(nF1,F1,s=0)
    tck_F2 = interpolate.splrep(nF2,F2,s=0)
    # tck_F3 = interpolate.splrep(nF3,F3,s=0)
    nF0new = np.arange(1, max(nF0) + 1, 10)
    nF1new = np.arange(1, max(nF1) + 1, 10)
    nF2new = np.arange(1, max(nF2) + 1, 10)
    # nF3new = np.arange(1, max(nF3) + 1, 10)
    F0_i = interpolate.splev(nF0new,tck_F0 ,der=0)
    F1_i = interpolate.splev(nF1new,tck_F1 ,der=0)
    F2_i = interpolate.splev(nF2new,tck_F2 ,der=0)
    # F3_i = interpolate.splev(nF3,tck_F3 ,der=0)


    pyplot.plot(nF0new, F0_i,'-r', nF1new, F1_i,'-b', nF2new, F2_i,'-k', lw= 1.5)
    pyplot.plot(nF0 , F0, 'ro', nF1, F1, 'bo', nF2, F2, 'ko')
    pyplot.grid(which='major', axis='both', linestyle='-',color='#C0C0C0', alpha=0.5)
    pyplot.grid(which='minor', axis='y', linestyle='--',color='#C0C0C0', alpha=0.5)
    pyplot.xlabel("n")
    # pyplot.ylim(ymin=0)
    pyplot.legend(('F0','F1', 'F2'), loc=2)
    pyplot.ylabel("Time (s)")
    pyplot.title("Execution Times - Serial (%d simulation runs) " % RUNS)
    pyplot.savefig('serial.pdf')
