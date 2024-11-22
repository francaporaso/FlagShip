import matplotlib.pyplot as plt
import numpy as np
# from vgcf import *

def plot_sky(ang_pos, rands_ang):
    
    plt.hist(ang_pos['ra'], bins=25, histtype='step', density=True)
    plt.hist(rands_ang['ra'], bins=25, histtype='step', density=True)
    plt.show()
    plt.hist(ang_pos['dec'], bins=25, histtype='step', density=True)
    plt.hist(rands_ang['dec'], bins=25, histtype='step', density=True)
    plt.show()
    plt.hist(ang_pos['z'], bins=25, histtype='step', density=True)
    plt.hist(rands_ang['z'], bins=25, histtype='step', density=True)
    plt.show()

    amin, amax = 0,90
    alpha = 0.9
    mran = (rands_ang['ra'] > amin) & (rands_ang['ra'] < amax) & (rands_ang['dec'] > amin) & (rands_ang['dec'] < amax)
    mtru = (ang_pos['ra'] > amin) & (ang_pos['ra'] < amax) & (ang_pos['dec'] > amin) & (ang_pos['dec'] < amax)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5))
    ax1.scatter(rands_ang['ra'][mran] * np.cos(rands_ang['dec'][mran]*np.pi/180), rands_ang['dec'][mran], color='green', s=0.1, alpha=alpha)
    ax1.set_xlabel('RA * cos(Dec)')
    ax1.set_ylabel('Dec')
    ax1.set_title('Randoms')

    ax2.scatter(ang_pos['ra'][mtru] * np.cos(ang_pos['dec'][mtru]*np.pi/180), ang_pos['dec'][mtru], color='blue', s=0.1, alpha=alpha)
    ax2.set_xlabel('RA * cos(Dec)')
    ax2.set_ylabel('Dec')
    ax2.set_title('Data')

    plt.show()

def plot_xyz(xyz_pos, rands_xyz):
    fig = plt.figure(figsize=plt.figaspect(0.5))
    ## plot data
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.scatter(*xyz_pos,
               s=1, alpha=0.3, c='b')
    ax.set_title('Data')
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.scatter(*rands_xyz,
               s=1, alpha=0.3, c='g')
    ax.set_title('Randoms')
    plt.show()