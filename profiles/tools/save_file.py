# guardar archivos
import astropy.io as fits

def save_file(*args):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(args[0], c='k')
    plt.show()