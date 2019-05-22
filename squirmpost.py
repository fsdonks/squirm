#given a set of results from a squirm optimization, derive the statistics we
#want to see.
from utils import *
import csv

#results are packaged in the form:
##{'totaldwelldev',
## 'locToDemand',
## 'locTostate',
## 'fillpath',
## 'flowcost',
## 'unitToHome',
## 'expectations',
## 'totalfilldev'}

#what we'd like to do is provide a suite of post processing tools that
#convert the result set (namely active flows) into useful information.
#Maybe we convert the active flow nodes into records like this:
#{"t" "unit" "location" "state" "demand"? "bogging"? "dwelling"?}



#return all the fills from the results set xs.
def getfills(xs):
    return map(first, xs["fillpath"])

def flowToRecord(kv):
    u,t,loc = kv
    return {"Unit": u, "Qtr" : t, "Location" : loc}

def getRecords(results):
    return map(flowToRecord, getfills(results))

#note -> it seems we have different schedules we can infer.
#we can grab a demand schedule, a supply (location->state) schedule,
#A readiness schedule, a bog/dwell schedule.
#Once we have the fills transformed into records, we can group them by
#unit, sort by time, and everything after that is transforming the data
#in the schedule.
def asTable(flows, rowkey=getKey("Unit"),colkey=getKey("Qtr")):
    schedule = []
    for u,uflows in groupby(rowkey,flows).iteritems():
        uflows.sort(key=colkey)
        timeline  =  [x for x in uflows if colkey(x) > 0]
        schedule.append((u, timeline))
    return schedule

def spitRecords(path,recs,fieldnames = None):
    with open(path, 'wb') as csvfile:
        if fieldnames == None:
            fieldnames = list(recs[0].keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(recs)


def postProc(res,interval=91):
    locstate = res["locTostate"]
    locdem   = res["locToDemand"]
    dmap     = res["dmap"]
    #filldev  = res["filldev"]
    #dwelldev = res["dwelldev"]
    recs = getRecords(res)
    expanded = []
    for r in recs:
        loc = r["Location"]
        #add on state, readiness, demand.
        if loc in locstate:
            r["State"] = locstate[loc]
        else:
            r["State"] = ""
        if loc in locdem:
            r["Demand"] = locdem[loc]
        else:
            r["Demand"] = ""
        t = (r["Qtr"] - 1) * interval
        if r["Demand"] in dmap:
            r["DemandGroup"] = dmap[r["Demand"]]["DemandGroup"]
        else:
            r["DemandGroup"] = r["State"]
        for i in range(interval):
            rnew = r.copy()
            rnew["t"] = i + t
            expanded.append(rnew)
    return expanded

#we just inverse the supplyschedule.
def demandSchedule(supplysched, locToDemand):
    pass

def readinessSchedule(supplysched, locToReadiness):
    pass

def homeSchedule(supplysched, unitToHome):
    pass


#the goal here is to build a table of data, which can then be post-processed
#and visualized.
def parseio(results):
    fpath = results["fillpath"]
    [flowToRecord(kv) for kv in fpath]
