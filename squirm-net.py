from itertools import product
from utils import *
import squirmsampledata as data
import squirmvis as vis
import squirmpost as post
import squirmnetwork as squirmnet
import pickle
from pulp import *

#setup some quick test data...we'll read this from actual data in the
#future, but for now we just embed it for testing.
def sampleData(dmax=14, smax=30):
    #substitutes define suitability.
    subs = {"I"  : {"ANY" : 0},
            "AI" : {"ANY" : 0},
            "S"  : {"ANY" : 0,
                    "I" : 1},
            "A"  : {"ANY" : 0,
                    "S"   : 2}}

    #default state transitions, basically 0-cost arcs.
    policyarcs = [["Ready",  "Ready"],
                  ["TransitionReady","Ready"],
                  ["TransitionPrepare", (1, "Prepare")],
                  [(1, "Prepare"),  "Prepare"],
                  ["Prepare", "Ready"]]

    #When we get to simulate the RC, this will matter
    #corresponds to a 1:4 RC policy
    rcpolicyarcs = [[(3, "Ready"), (2, "Ready")],
                    [(2, "Ready"), (1, "Ready")],
                    [(1,"Ready"), (0,"Ready")],
                    ["TransitionReady","Ready"],
                    ["TransitionPrepare", (1, "Prepare")],
                    [(1, "Prepare"),  "Prepare"],
                    ["Prepare", "Ready"]]

    #By convention, policy-start will be the hook-site for our transition
    #to demands.
    policystart = "Ready"
    #DemandTypes map to policies, which are subtrees of policy transitions.
    #we'll generate the appropriate arcs during parameter compilation.

    #A notional set of supply
    supply = [srecord(1,  "S", "AC" , 1,"A"),
              srecord(2,  "S",  "AC", 1,"A"),
              srecord(3,  "S",  "AC", 1,"B"),
              srecord(4,  "S",  "AC", 1,"D"),
              srecord(5,  "I",  "AC", 1,"D"),
              srecord(6,  "S",  "AC", 1,"E"),
              srecord(7,  "I",  "AC", 1,"F"),
              srecord(8,  "S",  "AC", 1,"G"),
              srecord(9,  "A",  "AC", 1,"G"),
              srecord(10, "A",  "AC", 1,"G"),
              srecord(11, "A",  "AC", 1,"H"),
              srecord(12, "A", "AC",  1,"H"),
              srecord(13, "A", "AC",  1,"H"),
              srecord(14, "S", "AC",  1,"H"),
              srecord(15, "A", "AC",  1,"I"),
              srecord(16, "A", "AC",  1,"I"),
              srecord(17, "S", "AC",  1,"J"),
              srecord(18, "A", "AC",  1,"J"),
              srecord(19, "I", "AC",  1,"J"),
              srecord(20, "A", "AC",  1,"K"),
              srecord(21, "I", "AC",  1,"K"),
              srecord(22, "I", "AC",  1,"M"),
              srecord(23, "I", "AC",  1,"M"),
              srecord(24, "I", "AC",  1,"N"),
              srecord(25, "AI", "AC", 1,"O"),
              srecord(26, "AI", "AC", 1,"O"),
              srecord(27, "AI", "AC", 1,"O"),
              srecord(28, "AI", "AC", 1,"P"),
              srecord(29, "AI", "AC", 1,"P"),
              srecord(30, "AI", "AC", 1,"P")]

    #sample demands with no overlap.
    demand = [drecord("D" + str(n), "ANY", "ANY",1, 1, 28, "COMMITTED", 0)
                        for n in range(7)]
    demand = demand + [drecord("D" + str(n), "ANY","ANY",1, 1, 28, "MISSION",0)
                                for n in range(8,14)]

    #allows us to truncate the data for smaller problem.
    supply = supply[0:smax]
    demand = demand[0:dmax]
    #just hack down the supply and demand to one for now.
    # supply = [supply[0]]
    # demand = [demand[0]]
    tmax = 28
    return {"supply"      : supply,
            "demand"      : demand,
            "subs"        : subs,
            "policyarcs"  : policyarcs,
            "policystart" : policystart,
            "tmax"        : tmax}

small = False

#setup input data for solving....
#ideally, we'd rip this from input files and provide the parameters
#necessary to solve.  That is left as an exercise for the reader...
def readInputData():
    global small
    if small:
        return sampleData(dmax=1,smax=5)
    else:
        return sampleData()

#data driven policy creation, so we can add new, freakier policies if we want.
defaultPolicies = {"SIMPLE"    : (1,"Transition", "Prepare"),
                   "COMMITTED" : (3,"Transition", "TransitionPrepare"),
                   "MISSION"   : (3,"Transition", "TransitionReady"),
                   ("RC", "MISSION")   : (3,"Transition", (3, "Prepare")),
                   ("RC", "COMMITTED") : (3,"Transition", (3, "Prepare"))}

#defines a sequence of transitions in which we wait.
def waitstate (name, length):
    return [(i,name) for i in range(length-1,0,-1)]

#makes a cycle of states.
def beside(xs):
    return [list(lr) for lr in pairwise(xs)]

#template for building generic SRM deploypolicies
def deploypolicy(name, length = 3, start = "Transition",
                 finish = "TransitionPrepare"):
    beginning = [(start, name)]
    end       = [(0, name),finish]
    return beside(beginning + waitstate(name,length) + end)

#given a demand record, what is its policy?
#Currently, we just dispatch based off the DEMANDTYPE in the record
def derivePolicy(name, demandtype):
    if demandtype in defaultPolicies:
        t,start,stop = defaultPolicies[demandtype]
        return deploypolicy(name,t,start,stop)
    else:
        raise Exception("unknown policy!" + str(demandtype)
                                          + "for " + str(name))
#Note-> we should chop this guy up into smaller functions.
#compile the intermediate data we want to use for our optimization.
#right now, we don't use subs.  The data is prepped, but not utilized.
def inputDataToParams(inputs):
    subs   = inputs["subs"]
    supply = inputs["supply"]
    demand = inputs["demand"]
    policyarcs  = inputs["policyarcs"]
    policystart = inputs["policystart"]
    #derive SRCs from the demands.  This is slightly hacky, but not terrible.
    srcs  = unions([distinctby(getKey("SRC"),demand),
                    distinctby(getKey("SRC"), supply)])
    dmap  = {r["NAME"] : r for r in demand}
    ds    =  dmap.keys()
    smap  = {r["NAME"] : r for r in supply}
    us    =  smap.keys()

    #derive policy arcs for the demands, and wire them up to ready.
    demandarcs = [derivePolicy(r["NAME"],r["DEMANDTYPE"]) for r in demand]
    #Define connections from policystart to demand.
    connections = [[policystart, ("Transition", d)] for d in ds]
    allarcs = reduce(lambda acc,x:acc + x, demandarcs, policyarcs + connections)
    #define our state transitions via implicit arcs, the key of the arc is
    #the cost.
    transitions = {(arc[0], arc[1]) : 0 for arc in allarcs}
    #note: right now, we're kind of blowing up the states.  we could
    #reformulate this to have fewer states but more constraints.
    #for now, I'll just leave it alone.

    #A map of demand names to either committed or mission.
    demandtype = {r["NAME"] : r["DEMANDTYPE"] for r in demand}
    #feasible states units my be in.
    #A unit is assigned a location, which corresponds to a state.
    locTostate = {}
    for (l,r),cost in transitions.iteritems():
        if not locTostate.has_key(l):
            if type(l) == tuple:
                if l[0] == "Transition":
                    locTostate[l] = demandtype[l[1]]
                elif l[1] == "Prepare" :
                    locTostate[l] ="Prepare"
                else:
                    locTostate[l] = demandtype[l[1]]
            else:
                locTostate[l] = l
        #ugly cut and paste, but meh
        if not locTostate.has_key(r):
            if type(r) == tuple:
                if l[0] == "Transition":
                    locTostate[r] = demandtype[r[1]]
                elif r[1] == "Prepare" :
                    locTostate[r] ="Prepare"
                else:
                    locTostate[r] = demandtype[r[1]]
            else:
                locTostate[r] = r
    #where u is from.
    unitTohome = {r["NAME"] : r["HOME"] for r in supply}
    homeSize      = {home : sum(map(lambda x : 1, units))
                     for home,units in itertools.groupby(supply,getKey("HOME"))}
    def asDemand(nd):
        if type(nd) == tuple:
            return nd[1]
        else:  return nd
    locToDemand = {}
    demandToLocs = {}
    for loc, state in locTostate.iteritems():
        d = asDemand(loc)
        if demandtype.has_key(d):
            locToDemand[loc] = d
            if demandToLocs.has_key(d):
                demandToLocs[d].append(loc)
            else:
                demandToLocs[d] = [loc]


    #build up our expectations over time, based on the demand schedule.
    #This is basically a table of all expected demand allocations over time.
    #Right now, each demand only expects one unit, per the data.
    expectations = {}
    for d in demand:
        name = d["NAME"]
        t    = d["START"]
        dur  = d["DURATION"]
        tend = t + dur
        for t in range(t,tend):
            expectations[(name,t)] = d["QTY"]
    inputs["srcs"] = srcs
    inputs["ds"] = ds
    inputs["us"] = us
    inputs["transitions"] = transitions
    inputs["expectations"] = expectations
    inputs["demandtype"] = demandtype
    inputs["locTostate"] = locTostate
    inputs["locToDemand"] = locToDemand
    inputs["demandToLocs"] = demandToLocs
    inputs["unitTohome"] = unitTohome
    inputs["homeSize"] = homeSize
    inputs["dmap"] = dmap
    inputs["smap"] = smap
    return inputs


#this helps recover our solution in an ordered fashion.
def comparenodes(l,r):
    u,t,l = l
    ru,rt,rl =r
    if t < rt: return -1
    elif t > rt: return 1
    else:
        if u < ru: return -1
        elif u > ru: return 1
        else:
            if l < rl: return -1
            elif l > rl: return 1
            else: return 0

def unitArcs(u,net):
    return filter(lambda (l,r) : l[0] == 0, net["arcs"])

#given a set of parameters, defines an instance of a TMAS model using
#the PuLP modelling functions.  As stated, PuLP will use its built in
#solver.
def buildModel(params):
    #bind parameters
    ds          = params["ds"]              #set of demand names
    us          = params["us"]              #set of units in supply
    unitcount   = len(us)                   #number of units in the supply
    srcs        = params["srcs"]            #set of srcs in supply
    tmax        = params["tmax"]            #max time horizon
    tmin        = 1
    transitions = params["transitions"]     #feasible transitions
    policystart = params["policystart"]     #starting node for any cycle (ready)
    locTostate  = params["locTostate"]      #mapping of locations to their
                                            #readiness state
    locToDemand  = params["locToDemand"]
    unitToHome   = params["unitTohome"]     #map of unit to origin.
    homeSize     = params["homeSize"]       #map of home to unit count
    demandToLocs = params["demandToLocs"]   #a map of demand to locations.
    ts           = range(tmin,tmax)         #the set of time indices.
    locs         = list(locTostate.keys())  #set of locations we can assign to.
    expectations = params["expectations"]   #set of quantity of expected fill,
                                            #by time (d,t)
    expecteddwell = {u : 8 for u in us}     #1:2 bog:dwell for ac goal for now.

    #index of (demand,time), note that itertools.product is only good for one
    #time, so if we want to keep it we need to dump it into a list.
    d_t      = list(product(ds,ts))

    print "building network"
    #given our parameters, we can define a network that encode the feasible
    #transitions over time.  This is a supply problem.
    net = squirmnet.buildNetwork(us,ts,transitions)
    nodes = net["nodes"]
    arcs  = net["arcs"]
    network = net["network"]
    SupplyNode = net["SupplyNode"]
    SupplyArc  = net["SupplyArc"]
    DemandNode = net["DemandNode"]
    DemandArc  = net["DemandArc"]
    sourcesOf  = net["sourcesOf"]
    sinksOf    = net["sinksOf"]
    capacity   = net["capacity"]
    cost       = net["cost"]
    flownode   = net["flownode"]
    S          = net["S"]
    T          = net["T"]
   # print network
    #flownodes are the interior nodes that we actually care about.
    flownodes = itertools.ifilter(flownode,nodes)
    demandnodes = set([(u,t,l)  for (u,t,l) in nodes if l in locToDemand])
    def sinks(nd):
        if nd in sinksOf:
            return sinksOf[nd]
        else:
            return []

    def sources(nd):
        if nd in sourcesOf:
            return sourcesOf[nd]
        else:
            return []
#The stuff before here should probably be lifted out into some supporting
#routines that build up the network, and plug in the necessary parameters
#for the model.  It's boilerplate atm.  Look into abstracting.
    #setting up a minimization.
    squirm = LpProblem("SquiRM", LpMinimize)

    #variables
    #Reformulating this as a network flow problem with side constraints.

    #The primary decisions we're making are flow-assignments. The difference
    #is that we're pushing flow between two nodes, node to individual states.
    #We can eliminate the need to check for readiness by simply not including
    #invalid arcs in the network.

    #unit readiness, demand fill, everything is function of where the units
    #are at a given point in time.  Location, therefore, is the central decision
    #to be made in the optimization.  We set up location as a binary variable.

    #Nodes exist across u,t,l , with feasible transitions between nodes
    #manifesting as arcs.  If we have flow going from (u,t,l) to
    # (u,t+1,r), then we know that assignments for (u,t,l) and (u,t+1,r)
    # took place.  We cap the flow at 1 to enforce only one assignment.

    #This lets us say "we want to create family of variables, across the
    #index u_t, with similar properties (in this case they're binary)
    #Note, we also get to lose the binary variable constraints.
    print "defining flow"
    flow = LpVariable.dicts('flow', arcs, lowBound= 0.0, cat=LpInteger)

    #For now, we have some notion of cost embedded in the network (although it's
    #zero by default).  If we define alternate state transitions, we can update
    #the cost to a non-zero value.  The point is, we are sillt minimizing cost
    #as a component of our flow model, so this is a mincost flow.
    flowcost = LpVariable('flowcost')
    squirm += flowcost == lpSum([flow[arc]*cost(arc) for arc in arcs])

    print "defining in/outflow"
    #helper variables: inflow(node) and outflow(node)
    inflow  = LpVariable.dicts('inflow', nodes)
    outflow = LpVariable.dicts('outflow', nodes)
    #Note: we need better error checking, the neighborhoods should be defined
    #for all nodes, i.e. no islands.
    for node in nodes:
        squirm += inflow[node]  == lpSum((flow[(source,node)]
                                            for source in sources(node)))
        squirm += outflow[node] == lpSum((flow[(node,sink)]
                                            for sink in sinks(node)))
##Flow Constraints
    #Default network flow constraints come in.  This is the "easy" part, since
    #most of our logical constraints are embedded in the structure of the
    #network now.

    print "defining flow conservation"
    #Flow conservation: inflow must be equal to outflow for every node.
    for node in nodes:
        if node != SupplyNode and node != DemandNode:
            squirm +=  inflow[node] == outflow[node]
    #Capacity: flow across an arc cannot exceed the capacity of the arc.
    for arc in arcs:
        squirm +=  flow[arc] <= capacity(arc)

    #For our assignment model, every entity must be assigned at a point in time.
    #This implies that we have to push all of our flow from the supply,
    #This means the flow to the DemandNode must be = to the capacity of supply,
    #in this case, the unit count.
    squirm +=  flow[DemandArc] == capacity(SupplyArc)

##Side-constraints and Variables.
    #Note, these are a side-problem that has nothing to do with the mincost
    #flow problem above.  If we can formulate cost in terms of the commodity
    #flowing across the network, we'd be done.  Our problem maps other costs
    #onto flow, so we have side-constrains and side-objectives to worry about.

    #What is fill?
    #Since nodes are defined (u,t,l), the fill at time t is equal to the inflow
    #of the node at time t.  We're basically tapping the arcs that we
    #care about, and summing their flow to determine classes of fill.  Note:
    #these are a byproduct of the network, and don't directly influence flow.
    #If we start to constrain flow by them, then we lose the properties of
    #netflow, and now have a mixed flow and lp to solve, which "may" be harder.

    #given a set of locations, we can determine how much fill we have on a
    #day by mapping the expected fill relative to the observed fill.  We observe
    #fill by summing across locations, so fill for a particluar demand
    #corresponds to the number of selections we have for a point in time across
    #all the locations that correspond to the demand.  We use demandToLocs to
    #perform the inverse mapping.
    print "defining fill"
    fill = LpVariable.dicts('fill', product(ds,ts))
    for d,t in product(ds,ts):
        squirm += fill[(d,t)] == lpSum([inflow[(u,t,l)]
                                                for u in us
                                                for l in demandToLocs[d]
                                                if (u,t,l) in demandnodes])

    #this all stays the same in the flow formulation.
    print "defining filldev"
    #Since we have a goal program, we'll allow fill to under-deviate, but not
    #over-deviate (for now?)

#is there another way to account for the single-assignment principle by
#summing flows?
#can we enforce linear constraints rather than leaning on an IP?


#    filldev = LpVariable.dicts("filldev", d_t)
    filldev = LpVariable.dicts("filldev", d_t, lowBound = 0)
    posdev  = LpVariable.dicts("posdev",  d_t, lowBound = 0)
    negdev  = LpVariable.dicts("negdev",  d_t, lowBound = 0)
    for d,t in product(ds,ts):
        squirm += filldev[(d,t)] == posdev[(d,t)] - negdev[(d,t)]

    print "relating fill to filldev"
    #fill deviation is the measure of  observed fill less the expected fill.
    #It may be positive or negative.
    for d,t in product(ds,ts):
        squirm += fill[(d,t)] + filldev[(d,t)] == expectations[(d,t)]

    #absolute fill deviation is the sum of our two positive variables, this
    #represents |filldev(d,t)|  We will use this in the objective.
    absfilldev = LpVariable.dicts("absfilldev", d_t)
    print "defining absfilldev"

    for d,t in product(ds,ts):
        squirm += absfilldev[(d,t)] == posdev[(d,t)] + negdev[(d,t)]

    #total deviation is sum of absolute deviation:
    print "defining totalfilldev"
    totalfilldev = LpVariable("totalfilldev")
    squirm += totalfilldev == lpSum([absfilldev[(d,t)]  for d in ds
                                                        for t in ts])
    #BOG:Dwell, MOB:Dwell
    #We can keep track of bog:dwell by summing over the course of a unit's
    #time horizon.
    print "defining bogdwell"
    dwell = LpVariable.dicts("dwell", us, lowBound = 0)
    bog   = LpVariable.dicts("bog"  , us, lowBound = 0)
    for u in us:
        squirm += dwell[u] == lpSum([inflow[nd] for nd in nodes
                                      if nd not in demandnodes and nd[0] == u])
        squirm += bog[u] == lpSum([inflow[nd] for nd in demandnodes
                                        if nd[0] == u])
    dwelldev     = LpVariable.dicts("dwelldev", us)
    dwellposdev  = LpVariable.dicts("dwellposdev",  us, lowBound = 0)
    dwellnegdev  = LpVariable.dicts("dwellnegdev",  us, lowBound = 0)

    for u in us:
        squirm += dwelldev[u] == dwellposdev[u] - dwellnegdev[u]

    #Given a bog:dwell goal, we can try to achieve it.
    #it's usually relative to 1:n, so we measure deviation from the
    #goal in terms of the proportion dwelldeviation.
    for u in us:
        squirm +=  dwell[u] + dwelldev[u]  == bog[u] / expecteddwell[u]

    #absolute fill deviation is the sum of our two positive variables, this
    #represents |filldev(d,t)|  We will use this in the objective.
    absdwelldev = LpVariable.dicts("absdwelldev", us)
    print "defining absdwelldev"
    for u in us:
        squirm += absdwelldev[u] == dwellposdev[u] + dwellnegdev[u]

    totaldwelldev = LpVariable("totaldwelldev")
    squirm += totaldwelldev == lpSum([absdwelldev[u] for u in us])

    #Home Drainage
    #One of the goals is to minimize the amount of time we have units leaving
    #home station simultaneously.  We can track this

    #For robustness, we'd like to enable alternate state transition paths that
    #have a cost associated with them (for instance, extending bog or somesuch).
    #In the network flow formulation, we handle this by weighting the cost of
    #flow along edges explicitly (faster/less clear/less convenient) or by
    #interpreting flow as cost in a different commodity via side-variables and
    #side-constraints.

    #objective - minimize the total fill deviation (for now)
    #note, the idiomatic way to do this is to use +=, but this is convenient for
    #doing hierarchical objective functions.

    #squirm.setObjective(totalfilldev)
    squirm.setObjective(totalfilldev + 0.1*totaldwelldev)

    def dictvals(d):
        return {k : value(d[k]) for k in d.keys()}

    def getsolution (debug=False):
        fillpath = []
        for nd in flownodes:
            if value(inflow[nd]) == 1.0:
                fillpath.append((nd, (value(inflow[nd]),value(outflow[nd]))))
        fillpath.sort(lambda l,r : comparenodes(l[0], r[0]))
        if debug:
            fills = [(SupplyNode, (value(inflow[SupplyNode]),
                      value(outflow[SupplyNode]))),
                     (S, (value(inflow[S]), value(outflow[S])))]    \
                     + fillpath
            for f in fills:
                print f
        return {#"transdev" : value(transdev),
                #"totalbad" : value(totalbad),
                "flow"          : dictvals(flow),
                "negdev"        : dictvals(negdev),
                "posdev"        : dictvals(posdev),
                "fill"          : dictvals(fill),
                "filldev"       : dictvals(filldev),
                "flowcost"      : value(flowcost),
                "totalfilldev"  : value(totalfilldev),
                "totaldwelldev" : value(totaldwelldev),
                "fillpath"      : fillpath,
                "locTostate"    : locTostate,
                "locToDemand"   : locToDemand,
                "unitToHome"    : unitToHome,
                "expectations"  : expectations,
                "dmap"          : inputs["dmap"],
                "smap"          : inputs["smap"]}
    return (squirm,getsolution) #return the model instance.

#We need to break this up.  Getting a bit large.

#keep track of our last solve, since this can take some time.
lastsolution = None
#Reads external data for parameters necessary to define TMAS, builds the model,
#solves the model, prints results.  Note: since we have Python data structures
#here, it should be really easy to pipe the results into post processing and
#other functions, using all of the Python facilities for operating on data.
def main(run = False):
    global lastsolution
    #should be replaced with IO functions to actual data
    if run:
        params = inputDataToParams(readInputData())
        squirm,getsolution = buildModel(params)
        print "Solving model...."
        squirm.solve()
        print "Status:", LpStatus[squirm.status]
        res = getsolution(debug=True)
        lastsolution = res
        return res

def spit(obj,path = "obj.py"):
    with open(path, 'wb') as outfile:
        pickle.dump(obj,outfile)

def load(path= "obj.py"):
    with open(path, "rb") as infile:
        return pickle.load(infile)

def writelp():
    params = inputDataToParams(readInputData())
    squirm,getsolution = buildModel(params)
    print "Saving model as SqurimModel.lp...."
    squirm.writeLP("SquirmModel.lp")


def restable(res):
    return post.asTable(post.getRecords(res))

def view(res):
    vis.plottrackstwo(restable(res))

if __name__ == '__main__': main()



