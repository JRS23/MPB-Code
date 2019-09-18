import math
import meep as mp
from meep import mpb
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CALCULATES BAND STRUCTURES FOR A SQUARE LATTICE OF SQUARE HOLES W/ A SOLID CORE, FOR SEVERAL VALUES OF KZ
# KZ: THE COMPONENT OF THE WAVE-VECTOR PARALLEL TO THE AXIS OF THE FIBER
# USER CREATES A LIST OF KZ VALUES BY SPECIFYING THE MAXIMUM AND MINIMUM VALUES AND THE INCREMENT SIZE (e.g. KZ_MIN = 0, KZ_MAX = 0.3, INCREMENT = 0.1 ====> KZ_LIST = {0, 0.1, 0.2, 0.3})
# SCALES LATTICE CONSTANT, SIDE-LENGTH OF HOLES, ETC., TO UNITS OF NANOMETERS
# PRINTS RELEVANT INFORMATION (E.G. NUMBER OF BANDS, PERMITTIVITIES OF MATERIALS, LATTICE CONSTANT, ETC.)
# DISPLAYS IMAGE OF FIBER CROSS-SECTION
# PLOTS ALL BAND STRUCTURES ON A 3D PLOT
# PLOTS THE OFF-AXIS ANGLES (THE OFF-AXIS ANGLE IS REFERRED TO AS 'THETA') CORRESPONDING TO THE LOWER AND UPPER BOUNDS OF BAND GAPS AS A FUNCTION OF FREQUENCY
# PLOTS FREQUENCY AS A FUNCTION OF THETA (THE PREVIOUS PLOT INVERTED)
# PLOTS THE ANGLE-GAPS AS A FUNCTION OF FREQUENCY
# PLOTS THE BAND STRUCTURE FOR A USER-CHOSEN VALUE OF KZ (KZ MUST BE IN THE LIST OF KZ DEFINED BY THE USER)
# AFTER THE PREVIOUS SIX PLOTS HAVE BEEN DISPLAYED, THE USER HAS THE CHOICE TO EITHER END THE PROGRAM OR GO BACK AND SEE THE PLOTS AGAIN

# OFF-AXIS ANGLES ARE COMPUTED FROM THE BAND STRUCTURE DATA VIA EQUATION (3-3) GIVEN IN "OFF-PLANE ANGLE DEPENDENCE OF PHOTONIC BAND GAP IN A TWO-DIMENSIONAL PHOTONIC CRYSTAL" (FENG & ARAKAWA)
# EFFECTIVE REFRACTIVE INDEX IS COMPUTED VIA EQUATION (3-2) GIVEN IN FENG & ARAKAWA
# THE EFFECTIVE REFRACTIVE INDEX WHEN THE CORE IS PRESENT IN THE PHOTONIC CRYSTAL IS GIVEN BY A MODIFIED VERSION OF THE FENG & ARAKAWA FORMULA
# BECAUSE THE CORE IS CENTRALLY-LOCATED IN THE FIBER, WE TREAT IT AS IF IT WERE SUBDIVIDED INTO EQUAL-SIZE SQUARE PIECES AND UNIFORMLY-DISTRIBUTED THROUGHOUT THE CROSS-SECTION OF THE FIBER

# State parameters of simulation: num_bands, epsilon, k_points, geometry, geometry_lattice, resolution

c = 299792458	# Speed of light (m/s)
n = math.pow(10, -7)	# One-tenth of a micron, or 10^(-7) meters, or 100 nm
billion = math.pow(10, 9)
N = n*billion	# math.pow(10, -7) * math.pow(10, 9) = 100

THETA_list = []		# THETA_list, frequencies, gap_boundaries, and THETA2_list will be filled with the relevant band structure data which will be plotted on diagrams at the end of the program
frequencies = []

gap_boundaries = []
THETA2_list = []

change = 'y'	# After inputting parameters, you'll be asked if you want to change any of them. If you say 'y', the program starts over.
while change == 'y':

	print()
	L = float(input("Enter the side length of the fiber: "))
	while L <= 0:
		L = float(input("Enter the side length of the fiber: "))

	LL = L/int(L)
	a = 2/LL

	print()
	ff = float(input("Enter the filling fraction of the square holes as a number between 0 and 1: "))  # Filling fraction of hole inside dielectric fiber; 0 < ff </= 1
	while ff <= 0:
		ff = float(input("Enter the filling fraction of the square holes as a number between 0 and 1: "))
	while ff >= 1:
		ff = float(input("Enter the filling fraction of the square holes as a number between 0 and 1: "))

	r = math.sqrt(ff) # Side length of square holes

	l_core = 3*(a/2)	# The side-length of the core is arbitrarily set equal to 1.5*a
	ff_core = math.pow(l_core, 2)/math.pow(L, 2)	# Ratio of core cross-sectional area to fiber cross-sectional area

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# SCALE r, l_core, L, AND a TO UNITS OF NANOMETERS

	r_physical = N*r
	l_core_physical = N*l_core
	L_physical = N*L
	a_physical = N*a

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------

	print()
	num_bands = int(round(float(input("Enter the number of bands as a positive integer: "))))   # Number of bands being calculated; a decimal will be rounded down to the nearest integer
	while num_bands <= 0:
		num_bands = int(round(float(input("Enter the number of bands as a positive integer: ")))) 

	print()
	core_epsilon = float(input("Enter the relative permittivity of the core as a number > 1: "))	# Relative permittivity of core material
	while core_epsilon <= 1:
		core_epsilon = float(input("Enter the relative permittivity of the core as a number > 1: "))

	print()
	pc_epsilon = float(input("Enter the relative permittivity of the photonic crystal material as a number > 1: "))	# Relative permittivity of photonic crystal material
	while pc_epsilon <= 1:
		pc_epsilon = float(input("Enter the relative permittivity of the photonic crystal material as a number > 1: "))

	contrast = pc_epsilon/core_epsilon	# Ratio of permittivities 

	pcepsilonstring = str(pc_epsilon)	# Used for printing the permittivity values on diagrams and graphs
	coreepsilonstring = str(core_epsilon)

	print()
	min_kz = float(input("Enter the lower bound on kz: "))
	while min_kz < 0:
		print("min_kz must be greater than or equal to zero.")
		print()
		min_kz = float(input("Enter the lower bound on kz: "))

	print()
	max_kz = float(input("Enter the upper bound on kz: "))
	while max_kz < min_kz:
		print("max_kz must be greater than or equal to min_kz.")
		print()
		max_kz = float(input("Enter the upper bound on kz: "))


	delta = max_kz - min_kz

	if delta > 0:
		print()
		step = float(input("Enter the step size for kz: "))
		while step <= 0:
			print("Step size must be positive.")
			print()
			step = float(input("Enter the step size for kz: "))
		denominator = int(1/step)
		kz_list = np.linspace(min_kz, max_kz, int(delta*denominator) + 1, endpoint=True) 	# List of kz values from min_kz to max_kz
		integer_list = range(0, len(kz_list), 1)  	# List of integers from 0 to the length of the kz_list

	else:
		kz_list = [max_kz]
		integer_list = range(0, len(kz_list), 1)
		step = 0
	
	print("min_kz = ", min_kz)
	print("max_kz = ", max_kz)
	print("step = ", step)
	print("kz_list: ", kz_list)
	print("integer_list: ", integer_list)

	
	block = input("Do you want to calculate with the core inserted? Type 'y' or 'n': ")
	while block != 'y' and block != 'n':
		block = input("Do you want to calculate with the core inserted? Type 'y' or 'n': ")

	if block == 'y':
		n_eff = math.sqrt(ff + ((1 - ff)*pc_epsilon) + ff_core*core_epsilon + ((1 - ff_core)*pc_epsilon))	# Modified Feng & Arakawa effective index formula to take into account the core

	else:
		n_eff = math.sqrt(ff + ((1 - ff)*pc_epsilon))	# Effective index defined in Feng & Arakawa
		
	constant = a/(2*math.pi*n_eff)	# Absorb all the constants in the Feng & Arakawa effective angle formula into a single constant

	kz_constants = []
	k_vectors = []

	for kz in kz_list:
		k_points = [mp.Vector3(0.0, 0.0, kz), mp.Vector3(0.5, 0.0, kz), mp.Vector3(0.5, 0.5, kz), mp.Vector3(0.0, 0.0, kz)] 
		k_points = mp.interpolate(10, k_points)	# Produces a list of 34 k-points around the irreducible Brillouin zone for the specified value of kz
		k_vectors.append(k_points)	# Each element of k_vectors is a list of 34 k-points; i.e. k_vectors is a list of lists
		
		q = kz*constant		# The Feng & Arakawa effective angle formula treats both kz and frequency (omega) as independent variables; by multiplying 'constant' by each value of kz, we create
					# a new list of constants so that, for each value of kz, the effective angle formula is now only a function of frequency.
		kz_constants.append(q)

	num_k = len(k_vectors)*len(k_vectors[0])
	print("Total number of wave-vectors = ", num_k)

	resolution = 32

	question = input("Do you want to use the targeted eigensolver? Enter 'y' or 'n': ")	# Targeted eigensolver solves for the num_bands bands nearest to the target frequency
												# WARNING: targeted eigensolver converges slowly and consequently takes much longer
	while question != 'y' and question != 'n':
		print()
		question = input("Do you want to use the targeted eigensolver? Enter 'y' or 'n': ")

	if question == 'y':	
		target = float(input("Enter the target frequency at which to compute the bands (THz): "))
		while target <= 0:
			print()
			print("Frequency cannot be zero or negative.")
			target = float(input("Enter the target frequency at which to compute the bands (THz): "))

		omega_prime = (target*math.pow(10, 12)*a_physical*math.pow(10, -9))/c	# Converts the physical frequency (in THz) into the dimensionless frequency variable often seen in band diagrams
		print("omega_prime = ", omega_prime)
		print("Target frequency = ", str(target) + " THz")

	print()
	change = input("Do you want to change any of the parameters? Type 'y' or 'n': ")
	while change != 'y' and change != 'n':
		change = input("Do you want to change any of the parameters? Type 'y' or 'n': ")

freqss = []
gapss = []
for index in integer_list:	# Loop over kz_list
	theta_list = []
	theta2_list = []

	geometry = []
	geometry_lattice = mp.Lattice(size=mp.Vector3(1, 1))
	ms = mpb.ModeSolver(num_bands=num_bands, k_points=k_vectors[index], geometry=geometry, geometry_lattice=geometry_lattice, resolution=resolution)	# Create ModeSolver object

	ms.default_material = mp.Medium(epsilon = pc_epsilon)	# Define background material to be that of the photonic crystal
	if question == 'y':
		ms.target_freq = omega_prime
	ms.geometry_lattice = mp.Lattice(size=mp.Vector3(L, L))	# Creates an L x L lattice
	ms.geometry = [mp.Block(material=mp.air, size=mp.Vector3(r,r,mp.inf))]	# One hole at each lattice point
	ms.geometry = mp.geometric_objects_lattice_duplicates(ms.geometry_lattice, ms.geometry)	# Duplicate geometry all over the lattice
	if block == 'y':
		ms.geometry.append(mp.Block(material=mp.Medium(epsilon=core_epsilon), size=mp.Vector3(l_core, l_core, mp.inf)))	# Inserts core into the center of the photonic crystal


	ms.run()

	freqs = ms.all_freqs
	freqss.append(freqs)
	gaps = ms.gap_list  # List of band gaps and their boundaries
	gapss.append(gaps)
	real_gaps = []	# List of gap-midgap ratios, expressed as percentages
	gap_bounds = []
	gap_bounds_freqs = []
	midgaps = []
	midgap_freqs = []
	num_gaps = len(gaps)  # Number of band gaps found by eigensolver
	false_positives = 0
	false_gaps = []
	threshold = 0.01  # If band gap (i.e. gap-midgap ratio) is less than the threshold, remove it from the list; it's probably a false positive
	numbers = range(num_gaps)


	if num_gaps == 0:
		real_gaps.append("Zero band gaps")
	else:
		for g in numbers:
			g = int(g) - 1
			if gaps[g][0] < threshold:	# If band-gap is smaller than threshold, we'll consider it to be a false-positive
				false_gaps.append(gaps[g])
				false_positives = false_positives + 1
			else:
				real_gaps.append(round(gaps[g][0], 3))  # List of band gaps
				gap_bounds.append((round(gaps[g][1], 3), round(gaps[g][2], 3)))  # List of boundaries of band gaps (omega_lower, omega_upper)
				midgaps.append(round(0.5*(gaps[g][1] + gaps[g][2]), 3))

	gapss_string = []	# Used for printing a list of the band-gaps at the end of the program, if the user so desires
	for gap in real_gaps:
		if num_gaps == 0:
			gapss_string.append("Zero band gaps")	
		else:
			gapss_string.append(str(gap) + "%")

	q = kz_constants[index]

	for omega in freqs[0]:
		if q/omega > 1:
			theta = 0
		else:
			theta = math.acos(q/omega)	# Feng & Arakawa effective angle formula, treating theta as a function of omega; q contains all of the constants and the value of kz
		theta_list.append(theta)
	
	THETA_list.append(theta_list)	# List of off-axis angles 

	#for midgap in range(len(midgaps)):
	#	midgap_freqs.append(midgaps[midgap]*(c/(a_physical*math.pow(10, -9))))
	#	midgap_freqs[midgap] = round(midgap_freqs[midgap]*math.pow(10, -12), 4)

	scaled_freqs = []

	for freq in range(len(freqs[0])):
		scaled_freqs.append(round(freqs[0][freq]*math.pow(10, -12)*(c/(a_physical*math.pow(10, -9))), 4))	# Translate dimensionless frequencies into physical values (THz)
	
	frequencies.append(scaled_freqs)	# Lists of frequencies (in THz)

	for bound in gap_bounds:
		new_bound = []
		for b in range(len(bound)):
			nb = round(bound[b]*math.pow(10, -12)*(c/(a_physical*math.pow(10, -9))), 4)	# Translate dimensionless frequencies into physical values (THz)
			new_bound.append(nb)

		gap_bounds_freqs.append(new_bound)

	gap_boundaries.append(gap_bounds_freqs)	# List of the boundaries of the band-gaps

	for bound in gap_bounds:
		theta3_list = []	
		for omega in bound:
			if q/omega > 1:
				theta = 0
			else:
				theta = math.acos(q/omega)
			theta3_list.append(theta)
		theta2_list.append(theta3_list)


	THETA2_list.append(theta2_list)	# Lists of the frequencies corresponding to the boundaries of the band-gaps

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
print()
print("min_kz = ", min_kz)
print("max_kz = ", max_kz)
print("step = ", step)
print("kz_list: ", kz_list)
print("integer_list: ", integer_list)
print()
print("Total number of wavevectors = ", num_k)
print("Side length of fiber = ", L)
print("Filling fraction of photonic crystal = ", ff)
print("Filling fraction of core in the fiber = ", round(ff_core, 3))
print("Permittivity contrast = ", round(contrast, 3))
print("Permittivity of photonic crystal material = ", pc_epsilon)
print("Relative permittivity of core = ", core_epsilon)
print("Effective refractive index of PBGF = ", round(n_eff, 4))
print("Number of bands calculated = ", num_bands)

if question == 'y':	# This information is only printed if you used the targeted eigensolver
	print("omega_prime = ", omega_prime)
	print("Target frequency = ", str(target) + " nm")

print()
print("Side length of core = " + str(l_core_physical) + " nm")
print("Side length of holes = " + str(round(r_physical, 4)) + " nm")
print("Cross-sectional side length = " + str(round(L_physical, 4)) + " nm")
print("Lattice constant = " + str(round(a_physical, 4)) + " nm")
print()
print("gap_boundaries: ", gap_boundaries)
print()
print("THETA2_list: ", THETA2_list)
print()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------

end = input("Do you want to end program? Type 'yes' or 'no': ")
if end != 'yes' and end != 'no':
	print()
	end = input("Do you want to end program? Type 'yes' or 'no': ")
	print()

while end == 'no':

# Draw photonic crystal cross-section
	p = 1	# Draws one period of the lattice geometry
	md = mpb.MPBData(rectify=True, periods=p, resolution=32)
	eps = ms.get_epsilon()
	converted_eps = md.convert(eps)
	plt.imshow(converted_eps.T, interpolation='spline36', cmap='binary')
	plt.axis('off')
	plt.show()


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------


	# PLOT ALL BAND STRUCTURES (ONE FOR EACH VALUE OF KZ) ON 3D PLOT

	if delta > 0:
		fig4 = plt.figure()
		ax4 = fig4.gca(projection='3d')

		x = range(len(freqss[0]))
		kz = kz_list
		X, KZ = np.meshgrid(x, kz)

	# Plot bands

		for index in integer_list: 
			ax4.plot(x, freqss[index], zs=kz[index], zdir='y', color='red', alpha=0.2)
	    
		ax4.set_xlim([x[0], x[-1]])	# x-axis goes from x[0] = 0 to x[-1] = 33; x[-1] = x[33], x[-2] = x[32], etc.
		ax4.set_ylim([0.0, max_kz])
		upper_z = freqss[-1][-1][-1] + 0.05
		ax4.set_zlim([0.0, upper_z])	# z-axis goes from 0 to upper_z

	# Plot labels
		if block == 'y':
			plt.title('Band Structure of Square Photonic Bandgap Fiber; $\epsilon$_core = ' + coreepsilonstring + '; $\epsilon$_PC = ' + pcepsilonstring, size=18)
		else:
			plt.title('Band Structure of Square Photonic Bandgap Fiber; $\epsilon$_PC = ' + pcepsilonstring, size=18)

		x_points_in_between = (len(freqss[0]) - 4) / 3
		x_tick_locs = [i*x_points_in_between+i for i in range(4)]
		x_tick_labs = ['Γ: (0,0)', 'X: (0.5, 0)', 'M: (0.5, 0.5)', 'Γ: (0,0)']
		ax4.set_xticks(x_tick_locs)
		ax4.set_xticklabels(x_tick_labs, size=7)

		delta = max_kz - min_kz
		y_tick_locs = [round((i/3)*delta, 2) for i in range(4)]
		y_tick_labs = [str(y_tick_locs[j]) for j in range(4)]
		ax4.set_yticks(y_tick_locs)
		ax4.set_yticklabels(y_tick_labs, size=7)

		z_tick_locs = [round((i/3)*upper_z, 2) for i in range(4)]
		z_tick_labs = [str(z_tick_locs[j]) for j in range(4)]
		ax4.set_zticks(z_tick_locs)
		ax4.set_zticklabels(z_tick_labs, size=7)

		ax4.xaxis.set_rotate_label(True)
		ax4.set_xlabel('Wavenumber: (kx, ky)', size=11)
		ax4.set_ylabel('kz', size=11)
		ax4.zaxis.set_rotate_label(False) 
		ax4.set_zlabel('Frequency ($\omega$a/2$\pi$c)', size=11, rotation=90)
		ax4.view_init(10, -40)
		ax4.grid(True)

		plt.show()


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------


# PLOT THETA AS A FUNCTION OF FREQUENCY

	fig1, ax1 = plt.subplots()

	ax1.set_ylim([THETA_list[len(THETA_list)-1][0]-0.08, THETA_list[0][-1]+0.08])
	ax1.set_xlim([frequencies[0][0] - 10, frequencies[len(frequencies)-1][-1] + 10])

	for integer in integer_list: 
		ax1.plot(frequencies[integer], THETA_list[integer], color='red')
		plt.plot(frequencies[integer], THETA_list[integer], 'bo')

	plt.axis([frequencies[0][0] - 10, frequencies[len(frequencies)-1][-1] + 10, THETA_list[len(THETA_list)-1][0]-0.08, THETA_list[0][-1]+0.08])

	plt.title('Off-Axis Angle vs. Frequency', size=18)

	ax1.set_xlabel('Frequency (THz)', size=16)
	ax1.set_ylabel('$\Theta$ (radians)', size=16)
	plt.show()


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# PLOT FREQUENCY AS A FUNCTION OF THETA; THIS IS JUST THE PREVIOUS DIAGRAM INVERTED

	fig2, ax2 = plt.subplots()

	ax2.set_xlim([THETA_list[len(THETA_list)-1][0]-0.08, THETA_list[0][-1]+0.08])
	ax2.set_ylim([frequencies[0][0] - 10, frequencies[len(frequencies)-1][-1] + 10])

	for integer in integer_list: 
		ax2.plot(THETA_list[integer], frequencies[integer], color='red')
		plt.plot(THETA_list[integer], frequencies[integer], 'bo')


	plt.axis([THETA_list[len(THETA_list)-1][0]-0.08, THETA_list[0][-1]+0.08, frequencies[0][0] - 10, frequencies[len(frequencies)-1][-1] + 10])

	plt.title('Frequency vs. Off-Axis Angle', size=18)

	ax2.set_xlabel('$\Theta$ (radians)', size=16)
	ax2.set_ylabel('Frequency (THz)', size=16)

	plt.show()


# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# PLOT ANGLE GAPS AS A FUNCTION OF FREQUENCY

	fig3, ax3 = plt.subplots()

	print()
	x_lower = float(input("Enter the lower bound of the x-axis (THz): "))
	while x_lower < 0:
		print()
		print("The lower bound cannot be negative.")
		print()
		x_lower = float(input("Enter the lower bound of the x-axis (THz): "))

	print()
	x_upper = float(input("Enter the upper bound of the x-axis (THz): "))
	while x_upper <= x_lower:
		print()
		print("The upper bound cannot be less than or equal to the lower bound.")
		print()
		x_upper = float(input("Enter the upper bound of the x-axis (THz): "))

	print()
	y_lower = float(input("Enter the lower bound of the y-axis (radians): "))
	while y_lower < 0 and y_lower >= math.pi/2:
		print()
		print("The lower bound cannot be negative or greater than or equal to pi/2.")
		print()
		y_lower = float(input("Enter the lower bound of the y-axis (radians): "))

	x_text = x_lower + (0.75*(x_upper - x_lower))

	ax3.set_xlim([x_lower, x_upper])
	ax3.set_ylim([y_lower, math.pi/2])

	for integer in integer_list: 
		for i in range(len(gap_boundaries[integer])):
			ax3.plot(gap_boundaries[integer][i], THETA2_list[integer][i], color='red')
			plt.plot(gap_boundaries[integer][i], THETA2_list[integer][i], 'bo')
			ax3.fill_between(gap_boundaries[integer][i], THETA2_list[integer][i][0], THETA2_list[integer][i][1], color='red', alpha=0.2)

	plt.axis([x_lower, x_upper, y_lower, math.pi/2])

	
	if block == 'y':
		plt.title('Off-Axis-Angle Gaps; $\epsilon$_core = ' + coreepsilonstring + '; $\epsilon$_PC = ' + pcepsilonstring, size=18)
		plt.text(x_text, y_lower+0.22, 'Core side-length = ' + str(round(l_core_physical, 3)) + " nm", size=15)
		plt.text(x_text, y_lower+0.20, 'Hole side-length = ' + str(round(r_physical, 3)) + " nm", size=15)

	else:
		plt.title('Off-Axis-Angle Gaps; $\epsilon$_PC = ' + pcepsilonstring, size=18)
		plt.text(x_text, y_lower+0.22, 'Hole side-length = ' + str(round(r_physical, 3)) + " nm", size=15)

	plt.text(x_text, y_lower+0.24, 'Lattice constant = ' + str(round(a_physical, 3)) + " nm", size=15)
	
	ax3.set_xlabel('Frequency (THz)', size=16)
	ax3.set_ylabel('$\Theta$ (radians)', size=16)
	plt.show()

# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------------------------------------------------

# PLOT A BAND DIAGRAM FOR A USER-CHOSEN VALUE OF KZ; KZ MUST BE IN THE KZ_LIST

	fig5, ax5 = plt.subplots()
	xx = range(len(freqss[0]))

	# Pick a value of kz from kz_list
	print()	
	print("kz_list: ", kz_list)
	print()
	Kz = float(input("Enter the value of kz for which you want to see the band diagram: "))
	if Kz < kz_list[0] or Kz > kz_list[-1]:
		print()
		Kz = float(input("Enter the value of kz for which you want to see the band diagram: "))
	
	if delta > 0:
		index = int((denominator*Kz)) - 1
	else:
		index = 0

	# Plot bands
	ax5.set_ylim([0, freqss[index][-1][-1] + 0.1])
	ax5.set_xlim([xx[0], xx[-1]])
	ax5.plot(freqss[index], color='red')

	# Plot gaps
	for gap in gapss[index]:
	    if gap[0] > 0.1:
	        ax5.fill_between(xx, gap[1], gap[2], color='red', alpha=0.2)	# Shades the angle-gap in red


	# Plot title and labels
	if block == 'y':
		plt.title('Band Structure of Square Photonic Bandgap Fiber; $\epsilon$_core = ' + coreepsilonstring + '; $\epsilon$_PC = ' + pcepsilonstring, size=18)
		plt.text(1.5, freqss[index][-1][-1]+0.06, 'Lattice constant = ' + str(round(a_physical, 3)) + " nm", size=14)
		plt.text(1.5, freqss[index][-1][-1]+0.04, 'Core side-length = ' + str(round(l_core_physical, 3)) + " nm", size=14)
		plt.text(1.5, freqss[index][-1][-1]+0.02, 'Hole side-length = ' + str(round(r_physical, 3)) + " nm", size=14)

	else:
		plt.title('Band Structure of Square Photonic Bandgap Fiber; $\epsilon$_PC = ' + pcepsilonstring, size=18)
		plt.text(1.5, freqss[index][-1][-1]+0.06, 'Lattice constant = ' + str(round(a_physical, 3)) + " nm", size=14)
		plt.text(1.5, freqss[index][-1][-1]+0.04, 'Hole side-length = ' + str(round(r_physical, 3)) + " nm", size=14)


	points_in_between = (len(freqss[0]) - 4) / 3
	tick_locs = [i*points_in_between+i for i in range(4)]
	tick_labs = ['Γ: (0,0)', 'X: (0.5, 0)', 'M: (0.5, 0.5)', 'Γ: (0,0)']
	ax5.set_xticks(tick_locs)
	ax5.set_xticklabels(tick_labs, size=16)
	ax5.set_xlabel('Wavenumber: (kx, ky); kz = ' + str(round(Kz, 3)), size=16)
	ax5.set_ylabel('Frequency ($\omega$a/2$\pi$c)', size=16)
	ax5.grid(True)

	plt.show()


	end = input("Do you want to end program? Type 'yes' or 'no': ")		# If you enter 'no', you'll get to see all the pretty diagrams and plots again!
	if end != 'yes' and end != 'no':
		print()
		end = input("Do you want to end program? Type 'yes' or 'no': ")
		print()

print()
print("min_kz = ", min_kz)
print("max_kz = ", max_kz)
print("step = ", step)
print("kz_list: ", kz_list)
print("integer_list: ", integer_list)
print()
print("Total number of wavevectors = ", num_k)
print("Side length of fiber = ", L)
print("Filling fraction of photonic crystal = ", ff)
print("Filling fraction of core in the fiber = ", round(ff_core, 3))
print("Permittivity contrast = ", round(contrast, 3))
print("Permittivity of photonic crystal material = ", pc_epsilon)
print("Relative permittivity of core = ", core_epsilon)
print("Effective refractive index of PBGF = ", round(n_eff, 4))
print("Number of bands calculated = ", num_bands)

if question == 'y':	# This information is only printed if you used the targeted eigensolver
	print("omega_prime = ", omega_prime)
	print("Target frequency = ", str(target) + " nm")

print()
print("Side length of core = " + str(l_core_physical) + " nm")
print("Side length of holes = " + str(round(r_physical, 4)) + " nm")
print("Cross-sectional side length = " + str(round(L_physical, 4)) + " nm")
print("Lattice constant = " + str(round(a_physical, 4)) + " nm")
print()




















