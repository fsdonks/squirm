#data vis:

import squirmsampledata
import squirmpost as post
import matplotlib.pyplot as plt
import numpy as np
import itertools
from utils import *


#post process these guys
#results are currently in the form (u,t,l) : flowinfo
#we need to map them to
#u1 [[l0....lmax]]
#u2 .....
samplesched = [[1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0],  #A
               [0, 0, 2, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 0],  #B
               [0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 0, 3, 0, 3]]  #C
def flowtime(flow):
    return flow[0][1]
def flowunit(flow):
    return flow[0][0]

results = post.getfills(squirmsampledata.sampleresults)

def asTimeLoc(flow):
    (u,t,l),flowinfo = flow
    return (t,l)

def asSchedule(flows):
    schedule = []
    for u,uflows in groupby(first,flows).iteritems():
        uflows.sort(key=second)
        timeline    =  [third(x) for x in uflows if second(x) > 0]
        schedule.append((u, timeline))
    return schedule

def scheduleToTracks(sched):
    return [v for s,v in sortby(first,sched)]

def schedule(data=samplesched):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_aspect(1)

    def avg(a, b):
        return (a + b) / 2.0

    for y, row in enumerate(data):
        for x, col in enumerate(row):
            x1 = [x, x+1]
            y1 = [0, 0]
            y2 = [1, 1]
            if col == 1:
                plt.fill_between(x1, y1, y2=y2, color='red')
                plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), "A",
                                            horizontalalignment='center',
                                            verticalalignment='center')
            if col == 2:
                plt.fill_between(x1, y1, y2=y2, color='orange')
                plt.text(avg(x1[0], x1[0]+1), avg(y1[0], y2[0]), "B",
                                            horizontalalignment='center',
                                            verticalalignment='center')
            if col == 3:
                plt.fill_between(x1, y1, y2=y2, color='yellow')
                plt.text(avg(x1[0], x1[0]+1), avg(y1[0], y2[0]), "C",
                                            horizontalalignment='center',
                                            verticalalignment='center')

    plt.ylim(1, 0)
    plt.show()

#hacking out some simple colors right now...
colormap = {('Transition', "MISSION")  : "cornflowerblue",
            "MISSION" : "royalblue",
            'TransitionReady' : "greenyellow",
            'Ready'    : "green",
            "COMMITTED" : "orange",
            ('Transition', "COMMITTED") : "goldenrod",
            "Prepare" : "yellow",
            (0, "Prepare") : "yellow",
            (1, "Prepare") : "yellow",
            "TransitionPrepare"   : "palegoldenrod"}

#this is a hack.
def getColor(loc):
    if type(loc) == dict:
        return getColor(loc["Location"])
    elif type(loc) == tuple:
        l,r = loc
        if loc == "Transition":
            if loc in colormap:
                return colormap[loc]
            else:
                return "red"
        else:
            return getColor(second(loc)) #ignore the time.
    else: #we have a normal transition.
        if loc in colormap:
            return colormap[loc]
        else:
            return "red"

def astext(state):
    return str(state)

def plottracks(tracks, stateToColor=getColor,stateTotext=astext):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_aspect(1)
    plt.xlabel = "Time (Quarters)"
    plt.ylabel = "Unit"
    plt.title  = "Unit Readiness By Quarter"
    #for each row in the tracks, plot the track.
    #keep track of the current track.
    ylim = 1
    for y, row in tracks:
        plotcell(y-1,row,stateToColor, stateTotext)
        ylim = ylim + 1
    #change this
    plt.ylim(0,len(tracks))
    plt.show()

def plottrackstwo(tracks, stateToColor=getColor,stateTotext=astext):
    #for each row in the tracks, plot the track.
    #keep track of the current track.
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_title  = "Unit Readiness By Quarter"
    ax1.set_xlabel("Time (Quarters)")
    ax1.set_ylabel("Unit")
    ax1.set_ylim([1,30.0])
    ax1.set_xlim([1,28.0])
    for y, row in tracks:
        plotcelltwo(ax1,y-1,row,stateToColor, stateTotext)
#    ticks = map(lambda n : n * 28.0, ax1.get_xticks())
#    ax1.set_xticklabels(ticks)
    ax1.set_xticks(range(0,29))
    ax1.set_yticks(range(1,31))
    #change this
    plt.show()

def plotcell(y,row,stateToColor,stateToText):
    def avg(a, b):
        return (a + b) / 2.0
    #col is really the state.
    for x,state in enumerate(row):
        x1 = [x*10, x*10+10]
        y1 = [y,y]
        y2 = [y + 1, y + 1]
        clr = stateToColor(state)
        plt.fill_between(x1,y1,y2=y2, color = clr)
##        plt.text(avg(x1[0], x1[1]), avg(y1[0], y2[0]), stateToText(state),
##                                 horizontalalignment= 'center',
##                                 verticalalignment  = 'center')


import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plotcelltwo(ax,y,row,stateToColor,stateToText):
    yscale = 1.0 #float(1.0/30.0)
    xscale = 1.0 #float(1.0/28.0)
   # width, height = 0.2, 0.5
    for x,state in enumerate(row):
##        if x % 2 > 0:
##            clr = "red"
##        else:
##            clr = "green"
        clr = stateToColor(state)
        ax.add_patch(patches.Rectangle(
                (xscale * float(x), yscale * float(y)),   # (x,y)
                xscale,          # width
                yscale,          # height
                fc = clr))

##def plotcelltwo(ax,y,row,stateToColor,stateToText):
##    yscale = float(1.0/30.0)
##    xscale = float(1.0/28.0)
##   # width, height = 0.2, 0.5
##    for x,state in enumerate(row):
####        if x % 2 > 0:
####            clr = "red"
####        else:
####            clr = "green"
##        clr = stateToColor(state)
##        ax.add_patch(patches.Rectangle(
##                (xscale * float(x), yscale * float(y)),   # (x,y)
##                xscale,          # width
##                yscale,          # height
##                fc = clr))


