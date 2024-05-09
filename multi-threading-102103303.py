#Chirag Singla - 102103303 - COE11
import numpy as np
import threading
import time
from tabulate import tabulate

def multiply_matrices(matrix1, matrix2, result):
    for i in range(len(matrix1)):
        result[i] = np.dot(matrix1[i], matrix2)

def run_with_threads(num_threads):
    threads = []
    start_time = time.time()
    for i in range(num_threads):
        thread = threading.Thread(target=multiply_matrices, args=(random_matrices.copy(), constant_matrix.copy(), result_matrices))
        thread.start()
        threads.append(thread)
    # Wait for all threads to finish
    for thread in threads:
        thread.join()
    # End time
    end_time = time.time()
    return end_time - start_time

# Parameters
num_matrices = 200
matrix_size = 3000
random_matrices = [np.random.rand(matrix_size, matrix_size) for _ in range(num_matrices)]
constant_matrix = np.random.rand(matrix_size, matrix_size)
result_matrices = [None] * num_matrices

# Table headers
headers = ["Number of threads", "Execution time (s)"]

# Data rows
data = []
for num_threads in range(1, 9):
    execution_time = run_with_threads(num_threads)
    data.append([num_threads, execution_time])


import matplotlib.pyplot as plt
num_threads = [entry[0] for entry in data]
execution_times = [entry[1] for entry in data]
plt.plot(num_threads, execution_times, marker='o')
plt.xlabel('Number of threads')
plt.ylabel('Execution time (s)')
plt.title('Execution Time vs. Number of Threads')
plt.grid(True)
plt.show()
print(tabulate(data, headers=headers, tablefmt="grid"))
