import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

def gaussian(mu, variance, color, mark):
    sigma = math.sqrt(variance)
    x1 = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    plt.plot(x1, stats.norm.pdf(x1, mu, sigma), mark, color=color)

# depth
gaussian(0.0555453,0.00828185, "red", "o")
gaussian(0.0493813,0.00651935, "blue", "o")
gaussian(0.0131082,0.00439129, "green", "o")
gaussian(-0.0375977,0.00818134, "yellow", "o")

# # width
# gaussian(-0.0623836,0.0989345, "red", "<")
# gaussian(-0.0298476,0.0721966, "blue", "<")
# gaussian(0.00469189,0.0401698, "green", "<")
# gaussian(-0.0375977,0.0307738, "yellow", "<")

# # height
# gaussian(0.487245,0.130964, "red", "^")
# gaussian(0.45164,0.100723, "blue", "^")
# gaussian(0.367856,0.0531909, "green", "^")
# gaussian(0.157265,0.0826135, "yellow", "^")


plt.show()