import time
import numpy as np
from pylab import *

from coffee.actions import Prototype

import logging
logging.getLogger('matplotlib').setLevel(logging.ERROR)

class Plotter(Prototype):
    """docstring for Plotter"""
    def __init__(self, system, frequency = 1, xlim = (-1,1), ylim = (-1,1), \
        findex = None, delay = 0.0, cc = 0.0):
        super(Plotter, self).__init__(frequency)
        if findex is not None:
            self.index = findex
            self.delay = delay
            self.cc    = cc
            self.system = system
            self.colors = ('b','g','r','c','m','y','k','orange') 
            from numpy import asarray
            ion()
            fig = figure(0,figsize=(10,10))
            ax = fig.add_subplot(111)
            self.lines = []

            # For real runs

            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\psi_0$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\psi_4$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\rho$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\rho'$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\sigma$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\sigma'$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label="I")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\mu$")))
            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label="u")))
            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label="v")))

            # For complex runs

            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$Re(\psi_0)$")))
            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$Im(\psi_0)$")))
            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$Re(\psi_4)$")))
            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$Im(\psi_4)$")))
            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$Re(I)$")))
            # self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$Im(I)$")))

            #self.lines = [ ax.add_line(Line2D(xlim,ylim)) for k in findex ] 
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)       
            ax.grid(True)
            self.axes = ax
            fig.show()
    
    def _doit(self, it, u):
        x = u.domain.meshes[0]
        index = np.asarray(self.index)
        f = u.data[index]
        mx = np.max(f.flat).real
        mn = np.min(f.flat).real
        self.axes.set_title("Iteration: %d, Time: %f" % (it,u.time))
        l = len(self.index)
        ioff()
        #C = 2.*abs(f[0]*f[1]) + 6.*(((abs(f[4]*f[5])) - (abs(f[2]*f[3])))**2)
        C = 2.*(f[0]*f[1]) + 6.*(f[4]*f[5] - f[2]*f[3] + self.cc)**2
        #C = abs(C)
        # mxC = np.max(C)
        #self.axes.set_ylim(0.0,max(mxC, mx),auto=True)
	    #self.axes.set_ylim(-5,5)     
        #

        # For real runs

        self.lines[0].set_xdata(x)
        self.lines[0].set_ydata(f[0].real)
        self.lines[0].set_color(self.colors[0])
        self.lines[1].set_xdata(x)
        self.lines[1].set_ydata(f[1].real)
        self.lines[1].set_color(self.colors[1])
        self.lines[2].set_xdata(x)
        self.lines[2].set_ydata(f[2].real)
        self.lines[2].set_color(self.colors[2])
        self.lines[3].set_xdata(x)
        self.lines[3].set_ydata(f[3].real)
        self.lines[3].set_color(self.colors[3])
        self.lines[4].set_xdata(x)
        self.lines[4].set_ydata(f[4].real)
        self.lines[4].set_color(self.colors[4])
        self.lines[5].set_xdata(x)
        self.lines[5].set_ydata(f[5].real)
        self.lines[5].set_color(self.colors[5])
        self.lines[6].set_xdata(x)
        self.lines[6].set_ydata(C.real)
        self.lines[6].set_color(self.colors[6])
        self.lines[7].set_xdata(x)
        self.lines[7].set_ydata(f[6].real)
        self.lines[7].set_color('silver')

        # self.lines[9].set_xdata(x)
        # self.lines[9].set_ydata(f[7].real)
        # self.lines[9].set_color(self.colors[0])
        # self.lines[10].set_xdata(x)
        # self.lines[10].set_ydata(f[8].real)
        # self.lines[10].set_color(self.colors[1])

        # For complex runs

        # self.lines[0].set_xdata(x)
        # self.lines[0].set_ydata(f[0].real)
        # self.lines[0].set_color(self.colors[0])
        # self.lines[1].set_xdata(x)
        # self.lines[1].set_ydata(f[0].imag)
        # self.lines[1].set_color(self.colors[1])
        # self.lines[2].set_xdata(x)
        # self.lines[2].set_ydata(f[1].real)
        # self.lines[2].set_color(self.colors[2])
        # self.lines[3].set_xdata(x)
        # self.lines[3].set_ydata(f[1].imag)
        # self.lines[3].set_color(self.colors[3])
        # self.lines[4].set_xdata(x)
        # self.lines[4].set_ydata(C.real)
        # self.lines[4].set_color(self.colors[4])
        # self.lines[5].set_xdata(x)
        # self.lines[5].set_ydata(C.real)
        # self.lines[5].set_color(self.colors[5])
        
        self.axes.legend(loc = 'lower center',prop={'size': 24},ncol=3)

        iteration = int(it / self.frequency)

        # path = '/mnt/Data/Woodard/Stream/'

        # if (iteration <=9):
        #     plt.savefig(path + 'Tslice_00000' + str(iteration) + '.png',format='png')
        # elif (iteration <= 99):
        #     plt.savefig(path + 'Tslice_0000' + str(iteration) + '.png',format='png')
        # elif (iteration <= 999):
        #     plt.savefig(path + 'Tslice_000' + str(iteration) + '.png',format='png')
        # elif (iteration <= 9999):
        #     plt.savefig(path + 'Tslice_00' + str(iteration) + '.png',format='png')
        # elif (iteration <= 99999):
        #     plt.savefig(path + 'Tslice_0' + str(iteration) + '.png',format='png')

        ion()
        plt.figure(0)
        draw()
        plt.pause(self.delay)
