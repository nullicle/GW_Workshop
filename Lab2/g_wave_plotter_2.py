#!/usr/bin/env python
# encoding: utf-8


"""
g_wave_plotter_2.py

Created by Chris Stevens 2023
Copyright (c) 2023 University of Canterbury. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, clear_output

from coffee.actions import Prototype

class Plotter(Prototype):
    def __init__(self, system, frequency = 1, xlim = (-1,1), ylim = (-1,1), \
        findex = None, labels = None, delay = 0.0):
        super(Plotter, self).__init__(frequency)
        if findex is not None:
            self.index = findex
            self.delay = delay
            self.system = system
            self.colors = ('b','g','r','c','m','y','k','orange') 
            self.xlim = xlim
            self.ylim = ylim
            self.labels = labels

            fig = plt.figure();
            ax = fig.add_subplot(1, 1, 1);
            self.axes = ax
            self.fig = fig
    
    def _doit(self, it, u):
        x = u.domain.meshes[0]
        index = np.asarray(self.index)
        f = u.data[index]

        self.axes.cla()
        self.axes.set_xlim(self.xlim[0], self.xlim[1])
        self.axes.set_ylim(self.ylim[0], self.ylim[1])
        self.axes.set_title("Iteration: %d, Time: %f" % (it,u.time))
        self.axes.grid()
        self.axes.set_xlabel("z",fontsize=14)
        for i in range(0,len(index)):
            self.axes.plot(x, f[i].real, self.colors[i], \
                            label = self.labels[i]);
        
        I = 2.*(f[0]*f[1]) + 6.*(f[4]*f[5] - f[2]*f[3])**2
        self.axes.plot(x, I.real, self.colors[len(index)], label="I")

        self.axes.legend(loc = 'lower center',prop={'size': 14},ncol=3)

        clear_output(wait = True)
        display(self.fig)
