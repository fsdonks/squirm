#-------------------------------------------------------------------------------
# Name:        network setup squirm
# Purpose:
#
# Author:      thomas.spoon
#
# Created:     13/04/2015
# Copyright:   (c) thomas.spoon 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------

from utils import *
#A problem came up, which we can detect: sometimes there are nodes that
#are either sources with no sink, or sinks with no source.  We'll have to
#deal with this when working on larger problems in the future....
def findislands(fromtos):
    print "findislands is a stub, ensure you don't have errors in your graph!"
    return []

#Given a set of (source,sink) directed arcs, returns a tuple of
#sources,sinks   , which are mappings of source->sinks, and sink->sources
#respectively.
def deriveNeighbors(fromtos):
    sources = {}
    sinks   = {}
    for source,sink in fromtos:
        if source in sinks:
            sinks[source].append(sink)
        else:
            sinks[source] = [sink]
        if sink in sources:
            sources[sink].append(source)
        else:
            sources[sink] = [source]
    return sources,sinks


#aux function to parametrically create our fill network, based off of the
#number of units, the amount of time.  Note that the bulk of the "work"
#is not in model formulation, but building the network topology and providing
#the parameters that feed the formulation.  The formulation is actually really
#simple.
def buildNetwork(us, ts, transitions):
    unitcount = len(us)
    tmax = max(ts)
   #we represent arcs in python as dictionaries of (from,to) : (cost, capacity)
    #flow-arcs - note, we're bounded now by the number of transitions.
    intermediatearcs     = {((u,t,l), (u,t+1,r)) : (cost,1)
                            for u in us
                            for t in ts
                            for (l,r),cost in transitions.iteritems()}
    #collect all the nodes in every arc into a nodeset.
    nodes = distinct(unzip(intermediatearcs.iterkeys()))
    #we also want to append a start-node to a flow-node
    SupplyNode = ("Supply",0,"Supply")
    S = ("S",0,"S") #source-node, bassically a distributor for all supply.
    T = ("T",tmax + 2,"T") #sink-node
    DemandNode = ("Demand",tmax + 3, "Demand") #convenience

    #This is the capcacitated arc that produces supply, should be = to
    #the quantity of supply we have on hand.
    SupplyArc = (SupplyNode, S)
    DemandArc = (T, DemandNode)
#   I realize after debugging that having a single supply node as folly.
#   We actual have multiple trans-shipment nodes: one for each unit.
#   These nodes are hooked up to the supply node, and they connect
#   to the nodes relative to their unit's index.  In other words, they
#   determine the (singular) flow for the whole unit, and form a class
#   in the graph.  THe current version works with one unit, but
#   if there are multiple units, their flow constraints don't affect eachother.
#   the solution is to add a layer that enforces the constraints.
    startarcs  = {}
    for u in us:
        unode =(u,0,"INIT")
        uarc = (S,unode) #uarcs connect the transhipment node with the units
        startarcs[uarc] = (0,1) #zero cost, single capacity assignments
        for unit,t,sink in nodes:
            if unit == u and t == 1:
                startarcs[(unode, (unit,t,sink))] = (0,1)

#    startarcs = {(S, (u,t,sink))   : (0,1) for u,t,sink   in nodes if t == 1}
    startarcs[SupplyArc] = (0,unitcount) #prepend the supply producer

    #this would be easier if we had a nice api for defining nodes.
    endarcs   =   {((u,t,source), T) : (0,1) for u,t,source in nodes
                                             if t == tmax + 1}
    endarcs[DemandArc] = (0,unitcount)

    #the network is a dictionary of (from,to): (cost capacity).
    network =   merge([startarcs,intermediatearcs,endarcs])

    #these are the capacitated arcs of the networks, just (node,node) pairs.
    arcs = network.keys()

    #Redefine our node indices over the complete network of arcs.
    #this would be better written using unzip.
    nodes = distinct(unzip(network.iterkeys()))
    #compute mappings of source->sinks, and sink->source, where source in nodes,
    #and sink in nodes
    sourcesOf,sinksOf = deriveNeighbors(arcs)
    #pending implementation.
    scopednodes = findislands(arcs)
        #helper functions for accessing fields in the network
    def capacity(nd):
        cost,cap = network[nd]
        return cap
    def cost(nd):
        cost,cap = network[nd]
        return cost
    #predicate to help us determine if a node is non-terminal,i.e. indicative
    #of an assignment.
    def flownode(nd):
        if nd == SupplyNode or nd == S or nd == T:
            return False
        else:
            return True
    return {"nodes"            : nodes,
            "arcs"             : arcs,
            "network"          : network,
            "SupplyNode"       : SupplyNode,
            "SupplyArc"        : SupplyArc,
            "DemandArc"        : DemandArc,
            "DemandNode"       : DemandNode,
            "sourcesOf"        : sourcesOf,
            "sinksOf"          : sinksOf,
            "capacity"         : capacity,
            "cost"             : cost,
            "flownode"         : flownode,
            "S"                : S,
            "T"                : T,
            "startarcs"        : startarcs,
            "endarcs"          : endarcs,
            "intermediatearcs" : intermediatearcs,
            "scopednodes"      : scopednodes}
