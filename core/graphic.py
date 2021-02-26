
import matplotlib.pyplot as plt

def plot_pls_exp_vs_pred(x, y, slope, intercept, line):
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=0, marker='s', label='Data points')
    ax.plot(x, intercept + slope * x, label=line)
    ax.set_xlabel('Ye')
    ax.set_ylabel('Yp')
    ax.legend(facecolor='white')
    plt.show()
    
