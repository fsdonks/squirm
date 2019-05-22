from itertools import product
from pulp import *

#helper function for converting a sequence of [fieldnames, values]
#to a set of records.
def asRecords(xs):
    fields = xs[0]
    def makerec(vals):
        return {fields[i] : vals[i] for i in range(fields.count)}
    return map(makerec,xs[0:end])

def srecord(name,src, compo, loc, qty):
    return {"NAME"  : name,
            "SRC"   : src,
            "COMPO" : compo,
            "QTY"   : qty,
            "LOC"   : loc}

def drecord(name, src, compo, qty, start, duration, dtype, overlap):
    return {"NAME"       : name,
            "SRC"        : src,
            "COMPO"      : compo,
            "QTY"        : qty,
            "START"      : start,
            "DURATION"   : duration,
            "DEMANDTYPE" : dtype,
            "OVERLAP"    : overlap}

def distinct(keyf,xs):
    return set(map(keyf,xs))
def rest(xs): return xs[1:]
def butlast(xs):
    return xs[:len(xs) - 1]

def getKey(k):
    return lambda x : x[k]

#setup some quick test data...we'll read this from actual data in the
#future, but for now we just embed it for testing.
def sampleData():
    subs = {"I"  : {"ANY" : 0},
            "AI" : {"ANY" : 0},
            "S"  : {"ANY" : 0,
                    "I" : 1},
            "A"  : {"ANY" : 0}}

    #default state transitions, basically 0-cost arcs.
    policyarcs = [["Ready",  "Ready"],
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
              srecord(6,  "S",  "RC", 1,"E"),
              srecord(7,  "I",  "AC", 1,"F"),
              srecord(8,  "S",  "AC", 1,"G"),
              srecord(9,  "A",  "AC", 1,"G"),
              srecord(10, "A",  "RC", 1,"G"),
              srecord(11, "A",  "RC", 1,"H"),
              srecord(12, "A", "AC",  1,"H"),
              srecord(13, "A", "AC",  1,"H"),
              srecord(14, "S", "AC",  1,"H"),
              srecord(15, "A", "AC",  1,"I"),
              srecord(16, "A", "AC",1,"I"),
              srecord(17, "S", "AC",1,"J"),
              srecord(18, "A", "AC",1,"J"),
              srecord(19, "I", "AC",1,"J"),
              srecord(20, "A", "AC",1,"K"),
              srecord(21, "I", "AC",1,"K"),
              srecord(22, "I", "AC",1,"M"),
              srecord(23, "I", "AC",1,"M"),
              srecord(24, "I", "AC",1,"N"),
              srecord(25, "AI", "AC",1,"O"),
              srecord(26, "AI", "AC",1,"O"),
              srecord(27, "AI", "AC",1,"O"),
              srecord(28, "AI", "AC",1,"P"),
              srecord(29, "AI", "AC",1,"P"),
              srecord(30, "AI", "AC",1,"P")]

    #sample demands with no overlap.
    demand = [drecord("D" + str(n), "ANY", "ANY",1, 1, 28, "COMMITTED", 0)
                        for n in range(7)]
    demand = demand + [drecord("D" + str(n), "ANY","ANY",1, 1, 28, "MISSION",0)
                                for n in range(8,14)]

    #just hack down the supply and demand to one for now.
    # supply = [supply[0]]
    # demand = [demand[0]]
    tmax = 10
    return {"supply"      : supply,
            "demand"      : demand,
            "subs"        : subs,
            "policyarcs"  : policyarcs,
            "policystart" : policystart,
            "tmax"        : tmax}

#setup input data for solving....
#ideally, we'd rip this from input files and provide the parameters
#necessary to solve.  That is left as an exercise for the reader...
def readInputData():
    return sampleData()

#given a demand record, what is its policy?
#Currently, we just dispatch based off the
#DEMANDTYPE in the record
def derivePolicy(name, demandtype):
    if demandtype == "MISSION" :
        return [[("Transition" , name), (2, name)],
                [(2, name), (1, name)],
                [(1, name), (0, name)],
                [(0, name), "TransitionReady"]]
    elif demandtype == "COMMITTED" :
        return [[("Transition", name), (2, name)],
                [(2, name), (1, name)],
                [(1, name), (0, name)],
                [(0, name), "TransitionPrepare"]]

#compile the intermediate data we want to use for our optimization.
#right now, we don't use subs.  The data is prepped, but not utilized.
def inputDataToParams(inputs):
    subs   = inputs["subs"]
    supply = inputs["supply"]
    demand = inputs["demand"]
    policyarcs  = inputs["policyarcs"]
    policystart = inputs["policystart"]
    #derive SRCs from the demands.
    srcs  =  distinct(getKey("SRC")  ,demand).union(distinct(getKey("SRC"), supply))
    ds    =  distinct(getKey("NAME") ,demand)
    us    =  distinct(getKey("NAME") ,supply)
    #derive policy arcs for the demands, and wire them up to ready.
    demandarcs = [derivePolicy(r["NAME"],r["DEMANDTYPE"]) for r in demand]
    #Define connections from policystart to demand.
    connections = [[policystart, ("Transition", d)] for d in ds]
    allarcs = reduce(lambda acc,x:acc + x, demandarcs, policyarcs + connections)
    #define our state transitions via implicit arcs
    transitions = {(arc[0], arc[1]) : 1 for arc in allarcs}
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
    #Right now, each demand only expects one unit.
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
    return inputs

#given a set of parameters, defines an instance of a TMAS model using
#the PuLP modelling functions.  As stated, PuLP will use its built in
#solver.

def buildModel(params):
    #bind parameters
    ds          = params["ds"]            #set of demand names
    us          = params["us"]            #set of units in supply
    srcs        = params["srcs"]          #set of srcs in supply
    tmax        = params["tmax"]          #max time horizon
    transitions = params["transitions"]   #feasible transitions
    policystart = params["policystart"]   #starting node for any cycle (ready)
    locTostate  = params["locTostate"]    #mapping of locations to their
                                          #readiness state
    locToDemand  = params["locToDemand"]
    demandToLocs = params["demandToLocs"] #a map of demand to locations.
    ts           = range(1,params["tmax"])   #the set of time indices.
    locs         = list(locTostate.keys())   #set of locations we can assign to.
    expectations = params["expectations"] #set of quantity of expected fill,
                                          #by time (d,t)
    print "building model"
    transcost = {}
    for l in locs:
        for r in locs:
            if transitions.has_key((l,r)): #locations are adjacent, feasible
                transcost[(l,r)] = 0
            else:
                transcost[(l,r)] = 100

    #index of  (unit,time,location)
    u_t_l     = [(u,t,l) for u in us
                         for t in ts
                         for l in locs]
    #index of (demand,time)
    d_t      = [(d,t) for d in ds
                      for t in ts]

    #setting up a minimization.
    squirm = LpProblem("SquiRM", LpMinimize)

    #variables


    #unit readiness, demand fill, everything is function of where the units
    #are at a given point in time.  Location, therefore, is the central decision
    #to be made in the optimization.  We set up location as a binary variable.

    #This lets us say "we want to create family of variables, across the
    #index u_t, with similar properties (in this case they're binary)
    print "defining select"
    select = LpVariable.dicts('select', u_t_l,
                                        lowBound = 0,
                                        upBound  = 1,
                                        cat = pulp.LpInteger)
                                            #Constraints
    #this looks like a bottleneck fyi.
    #We can only have a unit assigned to one location per time step:
    for t in ts:
        for u in us:
            squirm += lpSum([select[(u,t,l)] for l in locs]) == 1.0

    #given a set of locations, we can determine how much fill we have on a
    #day by mapping the expected fill relative to the observed fill.  We observe
    #fill by summing across locations, so fill for a particluar demand corresponds
    #to the number of selections we have for a point in time across all the
    #locations that correspond to the demand.  We use demandToLocs to perform
    #the inverse mapping.
    #fill = LpVariable.dicts('fill', d_t, lowBound = 0)
    print "defining fill"
    fill = LpVariable.dicts('fill', d_t)
    for t in ts:
        for d in ds:
            squirm += fill[(d,t)] == lpSum([select[(u,t,l)]
                                                for u in us
                                                for l in demandToLocs[d]])

    print "defining filldev"
    #Since we have a goal program, we'll allow fill to under-deviate, but not
    #over-deviate (for now?)
    filldev = LpVariable.dicts("filldev", d_t)
    posdev  = LpVariable.dicts("posdev",  d_t, lowBound = 0)
    negdev  = LpVariable.dicts("negdev",  d_t, lowBound = 0)
    for t in ts:
        for d in ds:
            squirm += filldev[(d,t)] == posdev[(d,t)] - negdev[(d,t)]
    print "relating fill to filldev"
    #fill deviation is the measure of  observed fill less the expected fill.
    #It may be positive or negative.
    for t in ts:
        for d in ds:
            squirm += fill[(d,t)] + filldev[(d,t)] == expectations[(d,t)]

    #absolute fill deviation is the sum of our two positive variables, this
    #represents |filldev(d,t)|  We will use this in the objective.
    absfilldev = LpVariable.dicts("absfilldev", d_t)
    print "defining absfilldev"
    for t in ts:
        for d in ds:
            squirm += absfilldev[(d,t)] == posdev[(d,t)] + negdev[(d,t)]

    #total deviation is sum of absolute deviation:
    print "defining totalfilldev"
    totalfilldev = LpVariable("totalfilldev")
    squirm += totalfilldev == lpSum([absfilldev[(d,t)]  for d in ds
                                                        for t in ts])

    #all pairs of state transitions not explicitly defined are illegal.
    #We want to prohibit these from occurring.
    #illegal_moves = [(l,r) for l,r in  product(locs,locs) if not
    #                 (l,r) in transitions]
    # u_t_badl_badr = [(u,t,l,r) for u in us
    #                            for t in butlast(ts)
    #                            for l,r in illegal_moves]

##Example of speeding up variable generation, although this may not be
##the problem, I may just be building too big of a model, too many binvars.
##    # ----- THIS IS THE SLOW LOOP ------ #
##
##    #inequality constraints
##    for r in range(m):
##        prob += pulp.lpSum(A[r][i]*x[i] for i in range(n) if A[r][i]) <= b[r]
##
##    # ---------------------------------- #
##    for r in range(m):
##        prob += pulp.LpAffineExpression((x[i], A[r][i]) for i in range(n)
##                                         if A[r][i]) <= b[r]

#Wow...this took a while to figure out.  If we want to make the pairs of moves
#illegal, we have to add constraints that make their sum <= 1, since they are
#binary and since either of them may occur in isolation.  Future reference:
#this is how you model a mutex in LP.  This is taking a long time to build,
#likely due to the size of illegalmoves. We're creating a ton of constraints
#here.
    print "defining mutexes"
    for u,t in product(us,butlast(ts)):
        for l,r in [(l,r) for l,r in  product(locs,locs) if not (l,r) in transitions]:
            squirm += select[(u,t,l)] + select[(u,t+1,r)] <= 1

    #For robustness, we'd like to enable alternate state transition paths that
    #have a cost associated with them (for instance, extending bog or somesuch).
    print "defining transition costs"
##    tcost = LpVariable("tcost")
##    squirm += tcost == lpSum([transcost[(l,r)] *
##                             (select[(u,t,l)] + select[(u,t+1,r)])
##                              for u in us
##                              for t in butlast(ts)
##                              for l in locs
##                              for r in locs])
    #objective - minimize the total fill deviation (for now)
    #note, the idiomatic way to do this is to use +=, but this is convenient for
    #doing hierarchical objective functions.
    squirm.setObjective(totalfilldev) #+ transcost, "solutioncost"

    def getsolution ():
        for u,t,l in u_t_l:
            if value(select[(u,t,l)]) > 0:
                print (u,t,l)
        print {#"transdev" : value(transdev),
               #"totalbad" : value(totalbad),
               "totalfilldev" : value(totalfilldev)}
            #print (u,t,l),value(select[(u,t,l)])
    return (squirm,getsolution) #return the model instance.

#Reads external data for parameters necessary to define TMAS, builds the model,
#solves the model, prints results.  Note: since we have Python data structures
#here, it should be really easy to pipe the results into post processing and
#other functions, using all of the Python facilities for operating on data.
def main():
    #should be replaced with IO functions to actual data
    params = inputDataToParams(readInputData())
    squirm,getsolution = buildModel(params)
    print "Solving model...."
    #squirm.writeLP("SquirmModel.lp")
    squirm.solve()
    print "Status:", LpStatus[squirm.status]
    getsolution()
##    for v in squirm.variables():
##        print squirm.name, "=", squirm.varValue

    # The optimised objective function value is printed to the screen
    print "Total Deviation ", value(squirm.objective)

if __name__ == '__main__':
    main()
