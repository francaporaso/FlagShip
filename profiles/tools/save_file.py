# guardar archivos
import astropy.io as fits

def save_file(*args):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(args[0], c='k', label='method1')
    plt.plot(args[1], c='r', label='method2')
    plt.plot(args[2], c='b', label='method3')

    plt.legend()
    plt.show()