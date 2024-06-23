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
        self.is_sorted = False


    def get_tracers(self, cat, RMAX=5.):
        square_distance = (cat.xhalo - self.xc)**2 +  (cat.yhalo - self.yc)**2 + (cat.zhalo - self.zc)**2
        mask = square_distance<=(RMAX*self.rv)**2

        cat = cat[mask] # ver si esto no es el problema! (idem a perfiles3d.py de codes_tesis/)

        for i in range(len(cat)):
            self.tr.append(Tracer(cat.xhalo[i], cat.yhalo[i], cat.zhalo[i], cat.lmhalo[i]))

    def sort_tracers(self):
        for i in range(len(self.tr)):
            self.tr[i].distanceto(self.xc, self.yc, self.zc)
        self.tr.sort(key = lambda x: x.d)
        self.is_sorted = True

    def radial_density_profile(self, cat, RMIN:float, RMAX:float, dr:float):

        # MeanDenTrac = 1 => Delta + 1 == Density
        MeanDenTrac = 1. # Numero de trazadores en el catalogo / volumen total del catalogo
        NBINS = int(round(((RMAX-RMIN)/dr),0))

        RMIN *= self.rv
        RMAX *= self.rv
        dr   *= self.rv

        Delta = np.zeros(NBINS)
        radius = RMIN
        i = 0

        self.get_tracers(cat=cat, RMAX=RMAX)
        self.sort_tracers()

        for n in range(NBINS):
            mass = 0.
            while (self.tr[i].d >= radius and self.tr[i].d < radius + dr):
                mass += 10.0**(self.tr[i].lm)
                i+=1

            volume = (4*np.pi/3)*((radius+dr)**3 - radius**3) # volumen del cascaron
            Delta[n] = (mass/volume)/MeanDenTrac - 1. 
            radius += dr

        return Delta