import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



def lineplot(filename, title, xlab, ylab, *curves):
    """ makes a line plot
        - title: plot title
        - filename: filename to use
        - xlab, ylab: axis labels
        - curves:  [([x1, x2, ...], name), ...]   = list of (data, name) tuples
    """
    lines, names = zip(*curves)
    for line in lines:
        plt.plot(range(len(line)), line)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.legend(list(names))
    plt.title(title)
    plt.savefig(filename)
    plt.close()