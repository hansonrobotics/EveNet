import json 
import csv
import pandas as pd
import numpy as np
import os 
import argparse

def converter(jsonfname,csvfname,ofile):
	f = open(jsonfname,'r')
	json_data = json.loads(f.read())
	csv_data = pd.read_csv(csvfname)
	json_time_gmt = []
	csv_time_array = []
	csv_new_data = []
	
	for i in range(len(json_data) - 1):
		json_time = np.fromstring(json_data[i]['start_time'][17:25], dtype = int, sep = ':')
		json_time[0] = json_time[0] + 3
		json_time_gmt.append(json_time)
	
	json_time_gmt = np.array(json_time_gmt)

	for i in range(len(csv_data) - 1):
		csv_time = np.fromstring(csv_data['time'][i][11:19], dtype = int, sep = ':')
		csv_time_array.append(csv_time)
	
	csv_time_array = np.array(csv_time_array)
	

	comp_indices = np.nonzero(np.in1d(csv_time_array,json_time_gmt))[0]

	
	for i in range(0,23):
		csv_new_data.append(csv_data[csv_data.columns.values[i]][comp_indices])
	
	with open(ofile, "w") as output:
	 writer = csv.writer(output, lineterminator='\n')
	 for i in csv_new_data:
	 	writer.writerow([i])



if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Converter', 
        formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('j',
        metavar='jsonfile',
        help = 'Abs Json file Directory')
	parser.add_argument('c',
        metavar='csvfile',
        help = 'Abs Csv  file Directory')
	args = parser.parse_args()
	jsonfname = args.j
	csvfname = args.c
	ofile = "output.csv"
	converter(jsonfname,csvfname,ofile)
