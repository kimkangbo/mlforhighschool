import matplotlib.pyplot as plt
import numpy as np

t = np.arange(0, 12, 0.01)
plt.plot(t)
plt.show()

y = np.sin(t)
plt.figure(figsize=(6,4))
#plt.plot(t, y)
plt.scatter(t, y)
plt.show()
