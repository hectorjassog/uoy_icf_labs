#read_hyades_mesh.py
"""
Will Trickey
22/11/18

Python script to interpret the mesh data for the hyades simulation of 
the fusion CDT ICF data labs

Need the ICF_data_labs_rcm.csv file
"""

#imports
import numpy as np
import matplotlib.pyplot as plt

#read the rcm.csv file
#open file
rcm_file = open('ICF_data_labs_rcm.csv','r')

#ignore header
rcm_file.readline()

#read in the data
data = []
for line in rcm_file:
	data.append([float(x) for x in line.split(',')])
		
rcm_file.close()

#convert data from file into numpy array
data_array=np.array(data)

#time in ns is first column
time = data_array[:,0]

#declare figure for plotting
fig1 = plt.figure(figsize=(20,15))

#for loop to plot each mesh line
for i in range(len(data_array[0])-1):
	plt.plot(time,data_array[:,i+1],'r')

#make graph look nice
plt.title("Lagrangian mesh plot of \nICF capsule compression",size=60)
plt.xlabel("Time (ns)",size =40)
plt.ylabel("Position (um)",size=40)
plt.axis([0,2.5,0,300])
plt.tick_params(labelsize=20)

#save graph as png
plt.savefig("mesh_plot.png",dpi=150,bbox_inches='tight')

plt.show()
