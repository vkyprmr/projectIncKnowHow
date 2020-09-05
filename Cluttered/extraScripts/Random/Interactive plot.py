# -*- coding: utf-8 -*-
"""
Created on Sat May 11 19:39:05 2019

@author: Vicky Parmar
"""

#Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from cleandata import CleanData

#Data
df = CleanData('Complete 2015-2018 Data v1.2.csv',useHDF=True).df

variables = ['T4532','L4532','T4531','L4531','F516','T515']
df = df[variables].dropna()

#df.plot()

df_0 = df[:1000]

#Interactive legends
fig, ax = plt.subplots()
ax.set_title('Comparing impact of variables on Targets')
line1, = ax.plot(df_0.index, df_0.iloc[:,0], lw=1, color='red', label='T4532')
line2, = ax.plot(df_0.index, df_0.iloc[:,1], lw=1, color='blue', label='L4532')
line3, = ax.plot(df_0.index, df_0.iloc[:,2], lw=1, color='green', label='T4531')
line4, = ax.plot(df_0.index, df_0.iloc[:,3], lw=1, color='yellow', label='L4531')
line5, = ax.plot(df_0.index, df_0.iloc[:,4], lw=1, color='magenta', label='F516')
line6, = ax.plot(df_0.index, df_0.iloc[:,5], lw=1, color='black', label='T515')
leg = ax.legend(loc='upper right', fancybox=False, shadow=False)
leg.get_frame().set_alpha(0.4)

lines = [line1, line2, line3, line4, line5, line6]
lined = dict()
for legline, origline in zip(leg.get_lines(), lines):
    legline.set_picker(5)  # 5 pts tolerance
    lined[legline] = origline
	
def onpick(event):
    # on the pick event, find the orig line corresponding to the
    # legend proxy line, and toggle the visibility
    legline = event.artist
    origline = lined[legline]
    vis = not origline.get_visible()
    origline.set_visible(vis)
    # Change the alpha on the line in the legend so we can see what lines
    # have been toggled
    if vis:
        legline.set_alpha(1.0)
    else:
        legline.set_alpha(0.2)
    fig.canvas.draw()

fig.canvas.mpl_connect('pick_event', onpick)

plt.show()


#Checkbox plot
from matplotlib.widgets import CheckButtons
fig, ax = plt.subplots()
ax.set_title('Comparing impact of variables on Targets')
line1, = ax.plot(df_0.index, df_0.iloc[:,0], visible=True, lw=1, color='red', label='T4532')
line2, = ax.plot(df_0.index, df_0.iloc[:,1], visible=True, lw=1, color='blue', label='L4532')
line3, = ax.plot(df_0.index, df_0.iloc[:,2], visible=False, lw=1, color='green', label='T4531')
line4, = ax.plot(df_0.index, df_0.iloc[:,3], visible=False, lw=1, color='yellow', label='L4531')
line5, = ax.plot(df_0.index, df_0.iloc[:,4], visible=False, lw=1, color='magenta', label='F516')
line6, = ax.plot(df_0.index, df_0.iloc[:,5], visible=False, lw=1, color='black', label='T515')
plt.subplots_adjust(left=0.2)

rax = plt.axes([0.05, 0.4, 0.1, 0.15])
check = CheckButtons(rax, ('T4532', 'L4532', 'T4531', 'L4531', 'F516', 'T515'), (True, True, False, False, False, False))

for i, c in enumerate(["r", "b", "g", 'y', 'm', 'k']):
    check.labels[i].set_color(c)
    check.labels[i].set_alpha(1)

plt.show()

def func(label):
    if label == 'T4532':
        line1.set_visible(not line1.get_visible())
    elif label == 'L4532':
        line2.set_visible(not line2.get_visible())
    elif label == 'T4531':
        line3.set_visible(not line3.get_visible())
    if label == 'L4531':
        line4.set_visible(not line4.get_visible())
    elif label == 'F516':
        line5.set_visible(not line5.get_visible())
    elif label == 'T515':
        line6.set_visible(not line6.get_visible())

    plt.draw()

check.on_clicked(func)

plt.show()

