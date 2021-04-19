from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Task 1
x = [1, 2, 3, 4, 5, 6, 7]
y = [3.5, 3.8, 4.2, 4.5, 5, 5.5, 7]

plt.plot(x, y)
plt.show()

plt.scatter(x, y)
plt.show()

# Task 2
t = np.linspace(0, 10, 51)
f = np.cos(t)

plt.plot(t, f, color='green')
plt.axis([0.5, 9.5, -2.5, 2.5])
plt.title('График f(t)')
plt.xlabel('Значения t')
plt.ylabel('Значения f')
plt.show()

# Task 3
x = np.linspace(-3, 3, 51)
y1 = x**2
y2 = 2 * x + 0.5
y3 = -3 * x - 1.5
y4 = np.sin(x)

fig, ax = plt.subplots(nrows=2, ncols=2)
ax1, ax2, ax3, ax4 = ax.flatten()

fig.set_size_inches(8, 6)
fig.subplots_adjust(wspace=0.3, hspace=0.3)

ax1.plot(x, y1)
ax1.set_title("График y1")
ax1.set_xlim([-5, 5])

ax2.plot(x, y2)
ax2.set_title("График y2")

ax3.plot(x, y3)
ax3.set_title("График y3")

ax4.plot(x, y4)
ax4.set_title("График y4")
plt.show()

# Task 4
plt.style.use('fivethirtyeight')
data = pd.read_csv('creditcard.csv', sep=',')
class_counts = data['Class'].value_counts()
class_counts.plot(kind="bar")
plt.show()
class_counts.plot(kind="bar", logy=True)
plt.show()

v1_0 = data.loc[data['Class'] == 0]['V1']
v1_1 = data.loc[data['Class'] == 1]['V1']

plt.hist(v1_0, bins=20, density=True, alpha=0.5, color='grey', label='Class 0')
plt.hist(v1_1, bins=20, density=True, alpha=0.5, color='red', label='Class 1')
plt.legend(prop={'size': 10})
plt.xlabel('Class')
plt.show()