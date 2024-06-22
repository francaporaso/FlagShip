import numpy as np

class Tracer:
    def __init__(self,x,y,z,lm):
        self.x = x
        self.y = y
        self.z = z
        self.lm = lm
        self.d = None # dist to a center (xc,yc,zc)

    def distanceto(self, xc, yc , zc):
        d2 = (self.x - xc)**2 + (self.y - yc)**2 + (self.z - zc)**2
        self.d = np.sqrt(d2)


class Void:
    def __init__(self,xc,yc,zc,rv):
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.rv = rv
        self.tr = [] #tracers

    def comoving_to_celestial(self):
        # delta = 
        # alpha =
        # redshift = astropy tiene una formula...
        pass

    def get_tracers(self, cat, RMAX=5.):
        square_distance = (cat.xhalo - self.xc)**2 +  (cat.yhalo - self.yc)**2 + (cat.zhalo - self.zc)**2
        mask = square_distance<=(RMAX*self.rv)**2

        cat = cat[mask] # ver si esto no es el problema! (idem a perfiles3d.py de codes_tesis/)

        for i in range(len(cat)):
            self.tr.append(Tracer(cat.xhalo[i], cat.yhalo[i], cat.zhalo[i], cat.lmhalo[i]))

    def sort_tracers(self):
        self.tr.sort(key = lambda x: x.d)

    def radial_density_profile(self, rmin, rmax, dr=0.5):

        # dr en Mpc, distancia entre cascarones

        # MeanNumTrac = 1 => Delta == Density
        MeanNumTrac = 1.

        shells = [] # acá defino todas las cascaras que se van a dividir
        for shell in shells:
            
            mass = 1 # masa del cascaron
            volume = 1 # volumen del cascaron
            Delta = (mass/volume)/MeanNumTrac - 1.