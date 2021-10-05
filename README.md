# Observed-mixing-Maud-Rise-Codes
Codes to reproduce science and figures of "Observed mixing at the flanks of Maud Rise in the Weddell Sea"
These codes were written to process data from NKE profiling float. 
Specifically, the codes are used and referenced in the "Observed mixing at the flanks of Maud Rise in the1
Weddell Sea" paper. This paper also includes information where to find the source files of the float data.

Download of the data and preparation
------------------------------------

To reproduce the computations and figures of the paper, you will first need to reproduce our folder structure:

0. Create a new folder "projectfolder" for this project, and inside create one folder called "codes" and one folder called "data"
1. Download the contents for the data folder from our repository. We included (float data-) files specific to our experiment, more data that is published by others needs to be downloaded in the next steps.
	* Run the script era5-download.py (e.g. python3 era5-download.py) to download some ERA5 data and put the files into projectfolder/data/era5
	* We also use bathymetry collocations, so you need to download the etopo bathymetry (https://www.ngdc.noaa.gov/mgg/global/) and place the file in projectfolder/data/etopo1.nc
	* Look for the collocations.py file and run it (e.g. python3 collocations.py). This will create collocations between the ERA5 data and float position data and output the results in pickled python objects for further processing. We included the intermediate results collocation_[float_identifier].nc into our data repository, so you might be asked to override them if you follow our descriptions to this point. 
	* For the additional materials of our paper and for figure 1, data from SOCCOM/Argo floats are used. Snapshots of this data can be found at SOCCOM float data Snapshot 2021-05-05 https://doi.org/10.6075/J0T43SZG. The data must be placed in /data/SOCCOM/[floatid]QC.nc We use the data of the following floats:
		⁻ 5905381
		- 5904468
		- 5904471
		- 5903616
		- 5905382

Now it is possible to reproduce the figures and analysis in the paper. 

Reproduction of the analysis and the figures
--------------------------------------------
The steps required to reproduce the results and figures of our study are collected in IPython notebooks. 
The notebooks are called projectfolder/codes/FigureX*.ipynb to reproduce the corresponding figure(s). Some additional computations and ressources can be found in projectfolder/codes/Paragraph_3.1_density_differences.ipynb and projectfolder/codes/oxygen,ipynb
The figures can be found in projectfolder/plots/figures/figure*.png

Questions
---------
For any questions about the data, reproducibillity or our publication, reach out to Martin Mohrmann (main author of these codes), Sebastiaan Swart or Céline Heuzé. 
