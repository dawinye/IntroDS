import numpy as np
import timeit
import matplotlib.pyplot as plt


def function1(array):
    return array ** 0.5

def function2(array):
    for a,c in enumerate(array):
        array[a] = array[a] ** 0.5

    return array



def q1():
    ##make a one dimensional array of size 1000, from 0-1
    

    ## run the efficient function 1000 times
    start_time1 = timeit.default_timer()
    for i in range(1000):
        g = np.random.random((1000,))
        function1(g)

    end_time1 = timeit.default_timer()

    start_time2 = timeit.default_timer()
    
    for i in range(1000):
        g = np.random.random((1000,))
        function2(g)
    end_time2 = timeit.default_timer()

    print("The time for function 1 to run was: ", (end_time1 - start_time1)/1000)

    print("The time for function 2 to run was: ", (end_time2 - start_time2)/1000)

    print("The ratio between vectorized operations and normal operations was 1:",round(((end_time2 - start_time2)/1000)/((end_time1 - start_time1)/1000),3))

def q2():
    
    g = np.random.normal(loc =  3, scale = 1,size = (1000,))
    print(round(np.mean(g),3), " ", round(np.std(g),3))
    h = g - 3
    print(round(np.mean(h),3), " ", round(np.std(h),3))
    print("1st quartile:", round(np.quantile(g**2,0.25),3), "2nd quartile:", round(np.quantile(g**2,0.5),3), "3rd quartile:", round(np.quantile(g**2,0.75),3))
    plt.boxplot(g**2)
    plt.show()

def q3():
    data = np.loadtxt("data.csv",skiprows = 1, usecols = np.arange(32), unpack = True, delimiter = ",")
    malx = [i for indx,i in enumerate(data[2]) if data[1][indx] == 1.0]
    maly = [i for indx,i in enumerate(data[3]) if data[1][indx] == 1.0]
    benx = [i for indx,i in enumerate(data[2]) if data[1][indx] == 0.0]
    beny = [i for indx,i in enumerate(data[3]) if data[1][indx] == 0.0]
    plt.scatter(malx,maly, c="red")
    plt.scatter(benx,beny, c="blue")
    plt.xlabel("Radius Mean")
    plt.ylabel("Texture Mean")
    plt.show()



def q4():
    data = np.loadtxt("data.csv",skiprows = 1, usecols = [col for col in range(0,32)], delimiter = ",")
    trueMean = np.mean(data[:,2:], axis = 0)
    means = []
    sample_size = len(data)
    for i in range(6):
        N = 10 * (2 ** i)
        distance = 0
        for j in range(10000):
            randomSample = np.random.choice(len(data),N, replace=True)
            avg = np.mean(data[randomSample], axis = 0)[2:]

            distance += np.linalg.norm(np.mean(data[:,2:], axis = 0) - avg)

        distance /= 10000
        means.append(distance)
        print(N, distance)
        
    
    for i in range(len(means)-2):
        print(means[i+2]/means[i])

