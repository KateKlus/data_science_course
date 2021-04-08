import numpy as np

a = np.array([[1, 6],
              [2, 8],
              [3, 11],
              [3, 10],
              [1, 7]])

# Среднее по каждому признаку (столбцу)
mean_a = np.mean(a, axis=0)
print("mean_a: " + str(mean_a))

# От а отнимем средние значения по столбцам
a_centered = a - mean_a
print("a_centered: \n" + str(a_centered))

# Cкалярное произведение столбцов массива a_centered
a_centered_sp = a_centered[0:, 0:1].reshape(1, 5)[0] @ a_centered[0:, 1:].reshape(1, 5)[0]

# Ковариация
print(a_centered_sp/(a.shape[0] - 1))

cov = np.cov(a.transpose())
print(cov)
