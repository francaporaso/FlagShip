import numpy as np

class Tracer:
    def __init__(self, x:float, y:float ,z:float , lm:float):
        self.x = x
        self.y = y
        self.z = z
        self.lm = lm
        self.d = None # dist to a center (xc,yc,zc)

    def distanceto(self, xc:float, yc:float , zc:float):
        d2 = (self.x - xc)**2 + (self.y - yc)**2 + (self.z - zc)**2
        self.d = np.sqrt(d2)


class Void:
    def __init__(self, xc:float, yc:float, zc:float, rv:float):
        self.xc = xc
        self.yc = yc
        self.zc = zc
        self.rv = rv
        self.tr = [] #tracers
        self.is_sorted = False


    def get_tracers(self, cat, RMAX:float, center:bool = False):
        distance = (cat.xhalo - self.xc)**2 +  (cat.yhalo - self.yc)**2 + (cat.zhalo - self.zc)**2
        mask = distance<=(RMAX*self.rv)**2

        cat = cat[mask]

        if not center:
            for i in range(len(cat)):
                t = Tracer(cat.xhalo[i], cat.yhalo[i], cat.zhalo[i], cat.lmhalo[i])
                t.distanceto(self.xc, self.yc, self.zc)
                self.tr.append(t)
        else:
            xhalo, yhalo, zhalo = cat.xhalo - self.xc, cat.yhalo - self.yc, cat.zhalo - self.zc
            xhalo /= self.rv 
            yhalo /= self.rv 
            zhalo /= self.rv 
            for i in range(len(cat)):
                t = Tracer(xhalo[i], yhalo[i], zhalo[i], cat.lmhalo[i])
                t.distanceto(0,0,0)
                self.tr.append(t)
        

    def sort_tracers(self):
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

        if not self.is_sorted or self.tr == []:
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
    
class StackedVoid(Void):
    def __init__(self, tracers):
        self.tr = tracers
        self.is_sorted = False

    def radial_density_profile(self, RMIN:float, RMAX:float, dr:float):

        # MeanDenTrac = 1 => Delta + 1 == Density
        MeanDenTrac = 1. # Numero de trazadores en el catalogo / volumen total del catalogo
        NBINS = int(round(((RMAX-RMIN)/dr),0))

        Delta = np.zeros(NBINS)
        radius = RMIN
        i = 0

        if not self.is_sorted:
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