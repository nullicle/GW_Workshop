#!/usr/bin/env python
# encoding: utf-8


"""
g_wave_plotter.py

Created by Chris Stevens 2023
Copyright (c) 2023 University of Canterbury. All rights reserved.
"""

import numpy as np
from pylab import *
import matplotlib
matplotlib.use('TkAgg')

from coffee.actions import Prototype

class Plotter(Prototype):
    def __init__(self, system, frequency = 1, xlim = (-1,1), ylim = (-1,1), \
        findex = None, delay = 0.0, cc = 0.0):
        super(Plotter, self).__init__(frequency)
        if findex is not None:
            self.index = findex
            self.delay = delay
            self.cc    = cc
            self.system = system
            self.colors = ('b','g','r','c','m','y','k','orange') 

            ion()
            fig = figure(0,figsize=(10,10))
            ax = fig.add_subplot(111)
            self.lines = []

            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\psi_0$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\psi_4$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\rho$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\rho'$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\sigma$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\sigma'$")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label="I")))
            self.lines.append(ax.add_line(Line2D(xlim,ylim, label=r"$\mu$")))

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)       
            ax.grid(True)
            self.axes = ax
            fig.show()
    
    def _doit(self, it, u):
        x = u.domain.meshes[0]
        index = np.asarray(self.index)
        f = u.data[index]
        self.axes.set_title("Iteration: %d, Time: %f" % (it,u.time))
        ioff()
        
        C = 2.*(f[0]*f[1]) + 6.*(f[4]*f[5] - f[2]*f[3] + self.cc)**2
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

        self.axes.legend(loc = 'lower center',prop={'size': 24},ncol=3)

        ion()
        plt.figure(0)
        draw()
        plt.pause(self.delay)
