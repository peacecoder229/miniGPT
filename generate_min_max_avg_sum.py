#!/usr/bin/env python3
import subprocess
import sys
import os
import time
import re
import signal
import pandas as pd
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import optparse
import json

def get_opts():
    """
    read user params
    :return: option object
    """
    parser = optparse.OptionParser()
    parser.add_option("--inputfile", dest="infile", help="provide Input file name", default=None)
    parser.add_option("--outputfile", dest="outfile", help="provide Output file name", default=None)
    parser.add_option("--rowrange", dest="range", help="provide row range as lo:hi", default=None)
    parser.add_option("--aggregatecolumns", dest="aggregate", help="provide newname : colomn name :stats wanted mean/max/min to be applied on the columns --aggregatecolumns=\"'cpuavgfreq': avg_mhz_[0-9]\\b|avg_mhz_1[0-5]\\b':mean\" ", default=None)

    (options, args) = parser.parse_args()
    return options



if __name__ == "__main__":
    options = get_opts()
    csv = options.infile
    outfile = options.outfile
    rows=options.range
    agregate=options.aggregate

    if(csv is None and outfile is None ):
        print("options not set right. Please check --help")
        sys.exit(1)
    data = dict()

    if outfile:
        fd = open(outfile, 'a')
    else:
        pass

    df = pd.read_csv(csv, sep=",", header=0)
    if rows:
        #print(df)
        lo, hi = rows.split(":")
        df = df[int(lo):int(hi)]
        #print(df)
    else:
        pass

    c = df.columns
    df[c] = df[c].apply(pd.to_numeric, errors='coerce')

    if agregate:
        outname, col, metric = agregate.split(":")
        val = eval("df[col].%s()" %(metric))
        val = val.round(2)
        if outfile == "stdout":
            print("%s,%s" %(outname, val))
        else:
            fd = open(outfile, 'a')
            fd.write("%s,%s\n" %(outname, val))
    else:

        for itm in c:
            data[itm] = list()
            #print(itm)
            #val = re.split(r"%s" %(varsep), c)
            avg = df[itm].mean()
            low = df[itm].min()
            high = df[itm].max()
            p25 = df[itm].quantile(0.25)
            p75 = df[itm].quantile(0.75)
            
            data[itm].extend([low, avg, p25, p75, high])   
        

        result_df = pd.DataFrame.from_dict(data, orient='index').transpose()
        #result_df['MetricType'] = ["mean", "P25", "P75"]
        result_df.index = ["Min", "mean", "P25", "P75" , "Max" ]
        result_df.index.name = "%Delta_stat"
        result_df = result_df.round(2)
        if outfile:
            result_df.to_csv(outfile, sep = ",")
        else:
            print(result_df)



