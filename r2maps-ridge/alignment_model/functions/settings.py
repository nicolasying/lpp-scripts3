####################### PARAMETERS #######################
"""
ROIs :
   -1. None
	0. IFGorb
	1. IFGtri
	2. TP
	3. TPJ
	4. aSTS
	5. pSTS

About Experiment : 51 subjects, 9 blocks per subject, around 300 scans per subject, 219486 voxels per scan.

 Subjects = [57, 58, 59, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 86, 
						87, 88, 89, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 103, 104, 105, 106, 108, 109, 110, 113, 114, 115]
"""
##########################################################
class paths:
	def __init__(self):
		# Paths
		self.rootpath = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/'
		self.path2Code = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/r2maps-ridge/'
		self.path2Data = '/volatile/r2maps-ridge/data'
		self.path2Output = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/r2maps-ridge'
		self.path2Figures = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018/MRI/lpp-scripts3/outputs/r2maps-ridge/figures'
		self.path2local = '/home/av256874/Documents'
		self.path2logs = '/neurospin/unicog/protocols/IRMf/LePetitPrince_Pallier_2018MRI/lpp-scripts3/r2maps-ridge/logs'

		
