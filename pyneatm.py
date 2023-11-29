#IMPORTING MODULS
from math import *
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")

#-----------------------------------------------------------------------

#Here we define some global constants: the Solar constant Ssolar, the
#Stefan--Boltzmann-constant σSB, the Astronomical unit in kilometers AU,
#the speed of light in vacuum clight, the Boltzmann-constant kB and the
#Planck-constant ħ.

S_solar = 1.367e+03 #[W m**-2]
sigma_sb = 5.670367e-08 #[W m**-2 K**-4]
au = 1.4959787e+08 #[km]
c_light = 2.99792458e+08 #[m s**-1]
k_b = 1.38064852e-23 #[m**2 kg s**-2 K**-1]
h_planck = 6.62607004e-34 #[m**2 kg s**-1]

#-----------------------------------------------------------------------

#We specify which quadrant of the plane a given angle is:

def quadrant(angle):
  if angle >= pi/2 and angle <= pi:
    return 1.0
  elif angle >= -pi and angle <= -pi/2:
    return -1.0
  else:
    return 0.0

Quadrant = np.vectorize(quadrant)

#Thus we can define an arcus tangent function with a return value 
#between [-pi, pi]:

def atanx(value, angle):
	return np.arctan(value) + pi*Quadrant(angle)

#-----------------------------------------------------------------------

#Planck function for a list of frequencies and a temperature

def B(nu, T):									
	toolarge = h_planck*nu/(k_b*T) < 100		#avoid overflow

	return toolarge.astype(int)*(2*h_planck*np.power(nu,3))/\
	(c_light**2 * (np.exp(h_planck*nu/k_b/T)-1))

#-----------------------------------------------------------------------
#For different instruments and filters, we have to calculate the color
#correction of the certain one. The cc(T,band) is responsible for this.


def cc(T, band):
	'''
	if band == 'irac4':
	  T0 = np.array([5000, 2000, 1500, 1000, 800, 600, 400, 200])
	  cc0 = np.array([1.0269, 1.0163, 0.0112, 1.0001, 0.9928, 0.9839, 0.9818, 1.1215])
	'''

	if band == 'mips24':
	  T0 = np.array([20000, 1000,  300,  150,  100,   80,   70,   60,   50,   40,   35,   30,   25,   20])
	  cc0 = np.array([1.000, 0.992, 0.970, 0.948, 0.947, 0.964, 0.986, 1.029, 1.119, 1.335, 1.569, 2.031, 3.144, 7.005])


	if band == 'mips70':
	  T0 = np.array([20000, 1000,  300,  150,  100,   80,   70,   60,   50,   40,   35,   30,   25,   20])
	  cc0 = np.array([1.000, 0.995, 0.980, 0.959, 0.938, 0.923, 0.914, 0.903, 0.893, 0.886, 0.888, 0.901, 0.941, 1.052])


	if band == 'mips160':
	  T0 = np.array([20000, 1000,  300,  150,  100,   80,   70,   60,   50,   40,   35,   30,   25,   20])
	  cc0 = np.array([1.000, 0.999, 0.996, 0.991, 0.986, 0.982, 0.979, 0.976, 0.971, 0.964, 0.959, 0.954, 0.948, 0.944])


	if band == 'pacs70':
	  T0  = np.array([10000.0, 5000.0, 1000.0,  500.0,  250.0,  100.0,   50.0,    40.0,   30.0,   20.0,   19.0,   18.0,   17.0,   16.0,   15.0,   14.0,   13.0,   12.0,   11.0,   10.0])
	  cc0 = np.array([1.06,  1.06, 1.013, 1.011, 1.005, 0.989, 0.982,  0.992, 1.034, 1.224, 1.269, 1.325, 1.396, 1.488, 1.607, 1.768, 1.992, 2.317, 2.816, 3.645])


	if band == 'pacs100':
	  T0  = np.array([10000.0, 5000.0, 1000.0,  500.0,  250.0,  100.0,   50.0,   40.0,   30.0,   20.0,   19.0,   18.0,   17.0,   16.0,   15.0,   14.0,   13.0,   12.0,   11.0,   10.0])
	  cc0 = np.array([1.034, 1.033, 1.031, 1.029, 1.023, 1.007, 0.985, 0.980, 0.982, 1.036, 1.051, 1.069, 1.093, 1.123, 1.162, 1.213, 1.282, 1.377, 1.512, 1.711])


	if band == 'pacs160':
	  T0  = np.array([10000.0, 5000.0, 1000.0,  500.0,  250.0,  100.0,   50.0,   40.0,   30.0,   20.0,   19.0,   18.0,   17.0,   16.0,   15.0,   14.0,   13.0,   12.0,   11.0,   10.0])
	  cc0 = np.array([1.704, 1.074, 1.072, 1.068, 1.062, 1.042, 1.010, 0.995, 0.976, 0.963, 0.964, 0.967, 0.972, 0.979, 0.990, 1.005, 1.028, 1.061, 1.110, 1.184])


	if band == 'spire250':
	  T0  = np.array([10000.0,  200.0,  100.0,   50.0,   20.0])
	  cc0 = np.array([1.058, 1.058, 1.047, 1.034, 0.998])


	if band == 'spire350':
	  T0  = np.array([10000.0,  200.0,  100.0,   50.0,   20.0])
	  cc0 = np.array([1.054, 1.054, 1.035, 1.028, 1.006])


	if band == 'spire500':
	  T0  = np.array([10000.0,  200.0,  100.0,   50.0,   20.0])
	  cc0 = np.array([1.060, 1.060, 1.060, 1.051, 1.025])


	if band == 'alma6':
	  T0  = np.array([10000.0,  10.0])
	  cc0 = np.array([1.0,  1.0])


	if band == 'alma7':
	  T0  = np.array([10000.0,  10.0])
	  cc0 = np.array([1.0,  1.0])


	if band == 'alma8':
	  T0  = np.array([10000.0, 10.0])
	  cc0 = np.array([1.0,  1.0])

	#CC = interpol(cc0,T0,T)
	CC = interpolate.interp1d(T0, cc0)
	return CC(T)

#-----------------------------------------------------------------------

#VECTOR CLASS
#we define a class for 3d vectors to calculate polyhedron grids

class vector3d:
	x = 0.0
	y = 0.0
	z = 0.0
	
	def __init__(self, coords = [0.0, 0.0, 0.0]):
		self.x = coords[0]
		self.y = coords[1]
		self.z = coords[2]
		
	def __add__(self, v):
		return vector3d([self.x+v.x, self.y+v.y, self.z+v.z])
		
	def __radd__(self, v):
		return self.__add__(v)
		
	def __sub__(self, v):
		return vector3d([self.x-v.x, self.y-v.y, self.z-v.z])
		
	def __mul__(self, v):
		if isinstance(v, self.__class__):
			return vector3d([self.y*v.z-self.z*v.y, self.z*v.x-self.x*v.z, self.x*v.y-self.y*v.x])
		if isinstance(v, float):
			return vector3d([self.x*v, self.y*v, self.z*v])	
			
	def __rmul__(self, v):
		return self.__mul__(v)
		
	def __pow__(self, v):
		return self.x*v.x + self.y*v.y + self.z*v.z
		
	def abs(self):
		return np.sqrt( np.power(self.x,2) + np.power(self.y,2) + np.power(self.z,2) )
		
	def __truediv__(self, v):
		return vector3d([self.x/v, self.y/v, self.z/v])
		
	def __str__(self):
		return "vector3d("+str(self.x)+","+str(self.y)+","+str(self.z)+")"

#-----------------------------------------------------------------------
def getdatafromstr(string):
	
	start1 = string.find(" = ")
	start2 = string.find("=")

	if start1 != -1:
		return string[start1+3:len(string)]
	elif start2 != -1:
		return string[start2+1:len(string)]
	else:
		return ""

#-----------------------------------------------------------------------

#Function to read polyhedron grid data from obj files

def readobj(filename):
	f = open(filename, "r")
	lines = f.readlines()
	
	vertices = []
	faces = []
	
	for line in lines:
		if line[0] == "#":
			continue
		if line[0:2] == "vt":
			continue
		if line[0:2] == "vn":
			continue 		
		if line[0:2] == "v ":
			line = line[2:-1]
			vertices.append( vector3d([float(i) for i in list(line.split(" "))]) )
		if line[0] == "f":
			line = line[2:-1]
			if line[-1] == " ":
				line = line[0:-2]
			faces.append( [ int(i.split("/")[0]) for i in list(line.split(" "))] )
	
	return (vertices, faces) #vertices: array of vector3d
							 #faces: array of arrays of indices of vertices in faces

#-----------------------------------------------------------------------

filepath = {
		"H4": "pixel_coords_map_nested_galactic_res4.fits",
		"H5": "pixel_coords_map_nested_galactic_res5.fits",
		"H6": "pixel_coords_map_nested_galactic_res6.fits",
		"cartesian": "cartesian",
		"polyhedron": "tetrahedron.obj"
		}

#The NEATM class contains every data and calculations for a given
#asteroid

class neatm:
	
	def __init__(self, parameters=None):
		
		self.polyhedron_source = ''
		self.CreatedSurfaceGrid = False
		self.ProjectedSurfaceGrid = False
		
		if parameters:
			f = open(parameters, "r")
		
		else:
			print("No parameters were given, cannot create neatm object.\n")
			quit()
			
		pars1 = f.readlines()
		
		f.close()
		
		
		pars = []
		
		for e in pars1:
			if e  != '\n' and e[0] != '#':
				pars.append(e[0:-1])
		
		for e in pars:
			if e[0:4] == 'name':
				self.name = getdatafromstr(e)
			
			if e[0:10] == 'healpixmod':
				self.healpixmod = getdatafromstr(e)
				
			if e[0:7] == 'r_helio':
				self.r_helio = float(getdatafromstr(e))
			
			if e[0:5] == 'r_geo':
				self.r_geo = float(getdatafromstr(e))
				self.r_geo_km = self.r_geo*au
			
			if e[0:5] == 'A_bol':
				self.A_bol = float(getdatafromstr(e))
			
			if e[0:3] == 'eta':
				self.eta = float(getdatafromstr(e))
				
			if e[0:2] == 'pV':
				self.pV = float(getdatafromstr(e))
			
			if e[0:2] == 'hv':
				self.hv = float(getdatafromstr(e))
			
			if e[0:7] == 'q_phase':
				self.q_phase = float(getdatafromstr(e))
			
			if e[0:4] == 'eps ' or e[0:4] == 'eps=':
				self.eps = float(getdatafromstr(e))
			
			if e[0:5] == 'scale':
				self.scale = float(getdatafromstr(e))
				self.diam = self.scale	
			
			if e[0:2] == 'a ' or e[0:2] == 'a=':
				self.a = float(getdatafromstr(e))
				print(self.a)
				
			if e[0:2] == 'b ' or e[0:2] == 'b=':
				self.b = float(getdatafromstr(e))
				
			if e[0:2] == 'c ' or e[0:2] == 'c=':
				self.c = float(getdatafromstr(e))
				
			if e[0:9] == 'theta_sun':
				self.theta_sun = float(getdatafromstr(e)) * pi/180
			
			if e[0:7] == 'phi_sun':
				self.phi_sun = float(getdatafromstr(e)) * pi/180
			
			if e[0:9] == 'theta_obs':
				self.theta_obs = float(getdatafromstr(e)) * pi/180
			
			if e[0:7] == 'phi_obs':
				self.phi_obs = float(getdatafromstr(e)) * pi/180
				
			if e[0:6] == 'nn_wvl':
				self.nn_wvl = int(getdatafromstr(e))
				
			if e[0:14] == 'eps_spec_wvls0':
				self.eps_spec_wvls0 = np.array([float(i) for i in list(getdatafromstr(e).split(","))])
				
			if e[0:13] == 'eps_spec_vals':
				self.eps_spec_vals = np.array([float(i) for i in list(getdatafromstr(e).split(","))])
				
			if e[0:17] == 'polyhedron source':
				self.polyhedron_source = getdatafromstr(e)
				filepath = {
				"H4": "pixel_coords_map_nested_galactic_res4.fits",
				"H5": "pixel_coords_map_nested_galactic_res5.fits",
				"H6": "pixel_coords_map_nested_galactic_res6.fits",
				"cartesian": "cartesian",
				"polyhedron": self.polyhedron_source
				}
		
		#---------------------------------------------------------------

		#NEATM model uses a certain grid, which can be either
		#HEALPIX 4, 5, 6, or Cartesian, the gridpoints of which
		#is stored in a textfile. The healpixmod variable in the NEATM class
		#contains the filepath to a grid.

		#HEALPIX GRID: H4, H5 H6 or cartesian

		
			  
		self.HPM = filepath.get(self.healpixmod)
		
		if self.name:
			print("\n\n >> new neatm object created, name:", self.name)
		else:
			print("\n\n >> new neatm object created, unnamed")
		
		if self.q_phase == 0.0:							#???
			self.q_phase = 0.479 + 0.336*self.pV
			self.A_bol = self.q_phase*self.pV
		
		self.calculated_f_mono = False
		self.calculated_f_ib = False
		
		#-----------------------------------------------------------------------
		
		#Tss is the subsolar temperature, which is the temperature of the point
		#from where the Sun can be seen in Zenith.

		self.Tss = (S_solar*(1-self.A_bol)/self.r_helio**2/self.eps/sigma_sb/self.eta)**0.25
		
		#calculating cartesian coordinates of the normal vector pointing to Earth
		x_obs = np.cos(self.theta_obs)*np.cos(self.phi_obs)
		y_obs = np.sin(self.theta_obs)*np.cos(self.phi_obs)
		z_obs = np.sin(self.phi_obs)
		
		self.n_obs = vector3d([x_obs, y_obs, z_obs])
		
		#-----------------------------------------------------------------------

		#The spectral emissivity εν function is given at eps_spec_wvls
		#wavelengths, its value is stored in the eps_spec_vals array. We need
		#the specrtal emissivity at other wavelengths, so we interpolate this
		#given function.
		
		#"desired wavelengths"
		self.wvls = np.zeros(self.nn_wvl)
		self.nu = np.zeros(self.nn_wvl)
		self.dnu = np.zeros(self.nn_wvl)

		for i in range(self.nn_wvl):
			self.wvls[i] = 10**(log10(1500.0)/(self.nn_wvl)*i) #nn_wvl-1 kellene
			self.nu[i] = c_light/(self.wvls[i]*1.0e-6)

		self.dnu[0] = self.nu[0]

		for i in range(1,self.nn_wvl):
			self.dnu[i] = abs(self.nu[i] - self.nu[i-1])

		Epsilon_Lambda_Interpolate = interpolate.interp1d(self.eps_spec_wvls0, self.eps_spec_vals)
		self.eps_spec = np.zeros(self.nn_wvl)
		self.eps_spec = Epsilon_Lambda_Interpolate(self.wvls)
		
		
	def help(self):
		print("\n\n >> ", self.name, " is a NEATM object that represents a Near-Earth asteroid.")
		print("Here are the different commands one can use in this model:", end='', sep='')
		#CREATE_GRID
		print("\n  >>> ", self.name, ".create_grid() : Creates a grid on the surface of the", end='', sep='')
		print(" asteroid, which can be either cartesian or HEALPix (4, 5 or 6). ", end='', sep='')
		print(" The healpixmod variable decides which grid is to be used (default: cartesian).", end='', sep='')
		#SET_HEALPIXMOD
		print("\n  >>> ", self.name, ".set_healpixmod(hpm) : Takes a string (hpm) as an argument,\n", end='', sep='')
		print("which can be \'H4\', \'H5\', \'H6\' or \'cartesian\'. Sets the healpixmod of the object.", end='', sep='')
		#PLOT_THETA_PHI
		print("\n  >>> ", self.name, ".plot_theta_phi() : Plots theta vs phi for each gridpoint.", end='', sep='')
		#SPECTRAL_EMISSIVITY
		print("\n  >>> ", self.name, ".spectral_emissivity(NN_WVL, EPS_SPEC_WVLS0, EPS_SPEC_VALS) : ", end='', sep='')
		print("With EPS_SPEC_VALS vs EPS_SPEC_WVLS0 numpy arrays, one can give the spectral emissivity", end='', sep='')
		print(" function of the body. Then the program interpolates this function to NN_WVL logarythmically", end='', sep='')
		print(" spaced points. EPS_SPEC_VALS and EPS_SPEC_WVLS0 must be the same size.", end='', sep='')
		#CALCULATE_T
		print("\n  >>> ", self.name, ".calculate_T() : Calculates the surface temperature of each gridpoint.", end='', sep='')
		#PROJECT_TO_FOCAL_GRID
		print("\n  >>> ", self.name, ".project_to_focal_grid(N) : Projects the surface grid to the", end='', sep='')
		print(" focal plane. With two rotations, the y axis of the coordinate system is pointing torwards the", end='', sep='')
		print(" observer. Then creates a N×N grid on the x-z plane and projects each surface grid point to it.", end='', sep='')
		#MONO_FLUX_DENSITY
		print("\n >>> ", self.name, ".mono_flux_density() : Integrates the monochromatic flux density for each", end='', sep='')
		print(" frequency for the whole focal grid.", end='', sep='')
		#IB_FLUX_DENSITY
		print("\n >>> ", self.name, ".ib_flux_density() : Integrates the in-band flux density for each", end='', sep='')
		print(" given instrument and corresponding center wavelengths for the whole focal grid.\n", end='', sep='')
		print(self.name, ".f_nu_ib stores the in-band flux densities with color correction, while ", end='', sep='')
		print(self.name, ".f_nu_mc stores the monochromatic flux densities without color correction, for the same wavelengths.\n", end='', sep='')
		
	def set_healpixmod(self, hpm):
		self.healpixmod = hpm
		if(hpm != "polyhedron"):
			self.HPM = filepath.get(hpm)
		
	def create_grid(self):
		#-----------------------------------------------------------------------

		#Here we create and then fill up arrays to store coordinates
		#for each gridpoint

		#hp: healpix
		#sn: surface normal
		#s: sphere, The difference between the direction of the Sun/observer and
		#surface normals on the corresponding sphere
		#n: normal, The difference between the direction of the Sun/observer and
		#surface normals on the ellipse surface
		
	
		if self.healpixmod != "cartesian" and self.healpixmod != "polyhedron":
			
			#IMPORTING HEALPIX DATA
			hdul = fits.open("/home/kinyeaux/unkp2022/python/HEALPixDir/"+self.HPM)

			self.hdr = np.array(hdul[0].header) # primary header

			self.data = np.array(hdul[1].data.tolist()).reshape(1, 2*hdul[1].data.size) # data table
			self.data = self.data.reshape(2,hdul[1].data.size, order = 'F')
			self.data_hdr = np.array(hdul[1].header) # data header
			
			hdul.close()
			
			self.hp_theta = np.zeros(self.data.size)			
			self.hp_phi = np.zeros(self.data.size)

			self.sn_theta = np.zeros(self.data.size)		
			self.sn_phi = np.zeros(self.data.size)
			self.sigma_s = np.zeros(self.data.size)			
			self.sigma_s_obs = np.zeros(self.data.size)
			self.sigma_n = np.zeros(self.data.size)		
			self.sigma_n_obs = np.zeros(self.data.size)
			
			#Calculation for healpix grids
			
			self.hp_theta = self.data[:][0] * pi/180
			self.hp_phi = self.data[:][1] * pi/180

			self.sn_theta = atanx((self.a/self.b)*np.tan(self.hp_theta), self.hp_theta)
			self.sn_phi = atanx((self.a*self.b/self.c)*np.tan(self.hp_phi)/((self.a**2)*(np.sin(self.hp_theta))**2 + (self.b**2)*(np.cos(self.hp_theta))**2)**0.5, self.hp_phi)

			self.sigma_s = np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_sun) + np.cos(self.hp_phi)*np.cos(self.phi_sun)*np.cos(self.hp_theta-self.theta_sun))
			self.sigma_n = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_sun) + np.cos(self.sn_phi)*np.cos(self.phi_sun)*np.cos(self.sn_theta-self.theta_sun))

			self.sigma_s_obs = np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_obs) + np.cos(self.hp_phi)*np.cos(self.phi_obs)*np.cos(self.hp_theta-self.theta_obs))
			self.sigma_n_obs = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_obs) + np.cos(self.sn_phi)*np.cos(self.phi_obs)*np.cos(self.sn_theta-self.theta_obs))

			dxy = np.arccos(np.cos(self.hp_theta)*np.cos(self.hp_phi))
					
		
		if self.healpixmod == "cartesian":
			self.n_theta_cartesian = 360
			self.n_phi_cartesian = 180
			self.dtheta = 360*pi/180 / self.n_theta_cartesian
			self.dphi = 180*pi/180 / self.n_phi_cartesian
			
			self.hp_theta, self.hp_phi = np.meshgrid(np.linspace(-180, 180, self.n_theta_cartesian) * pi/180, np.linspace(-90, 90, self.n_phi_cartesian) * pi/180)

			self.sn_theta = atanx((self.a/self.b)*np.tan(self.hp_theta), self.hp_theta)
			self.sn_phi = atanx((self.a*self.b/self.c)*np.tan(self.hp_phi)/((self.a**2)*(np.sin(self.hp_theta))**2 + (self.b**2)*(np.cos(self.hp_theta))**2)**0.5, self.hp_phi)

			self.sn_phi[self.sn_phi > pi] = self.sn_phi[self.sn_phi > pi] - pi
			self.sn_phi[self.sn_phi < -pi] = self.sn_phi[self.sn_phi < -pi] + pi

			self.sigma_s = np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_sun) + np.cos(self.hp_phi)*np.cos(self.phi_sun)*np.cos(self.hp_theta-self.theta_sun))
			self.sigma_n = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_sun) + np.cos(self.sn_phi)*np.cos(self.phi_sun)*np.cos(self.sn_theta-self.theta_sun))

			self.sigma_s_obs =  np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_obs) + np.cos(self.hp_phi)*np.cos(self.phi_obs)*np.cos(self.hp_theta-self.theta_obs))
			self.sigma_n_obs = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_obs) + np.cos(self.sn_phi)*np.cos(self.phi_obs)*np.cos(self.sn_theta-self.theta_obs))
			self.sn_theta = atanx((self.a/self.b)*np.tan(self.hp_theta), self.hp_theta)
			self.sn_phi = atanx((self.a*self.b/self.c)*np.tan(self.hp_phi)/((self.a**2)*(np.sin(self.hp_theta))**2 + (self.b**2)*(np.cos(self.hp_theta))**2)**0.5, self.hp_phi)

			self.sn_phi[self.sn_phi > pi] = self.sn_phi[self.sn_phi > pi] - pi
			self.sn_phi[self.sn_phi < -pi] = self.sn_phi[self.sn_phi < -pi] + pi

			self.sigma_s = np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_sun) + np.cos(self.hp_phi)*np.cos(self.phi_sun)*np.cos(self.hp_theta-self.theta_sun))
			self.sigma_n = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_sun) + np.cos(self.sn_phi)*np.cos(self.phi_sun)*np.cos(self.sn_theta-self.theta_sun))

			self.sigma_s_obs = np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_obs) + np.cos(self.hp_phi)*np.cos(self.phi_obs)*np.cos(self.hp_theta-self.theta_obs))
			self.sigma_n_obs = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_obs) + np.cos(self.sn_phi)*np.cos(self.phi_obs)*np.cos(self.sn_theta-self.theta_obs))
			
		if self.healpixmod == "polyhedron":
			self.vertices, self.faces = readobj(self.polyhedron_source)
			
			#setting max length of vectors to 1
			max_length = max([g.abs() for g in self.vertices])
			self.vertices = [g/max_length for g in self.vertices]

			self.gridpoints = []
			self.surface_normals = []

			#calculating position vectors of faces and face normal vectors
			for face in tqdm(self.faces, desc="Calculating gridpoint coordinates", bar_format='{l_bar:1}{bar:100}', ascii=' >#'):
				self.gridpoints.append((1/3)*(self.vertices[face[0]-1] + self.vertices[face[1]-1] + self.vertices[face[2]-1]))
				self.surface_normals.append((1/2)*(self.vertices[face[2]-1] - self.vertices[face[0]-1]) * (self.vertices[face[2]-1] - self.vertices[face[1]-1]))
			#self.gridpoints_X = self.gridpoints_X/max_length
			#self.gridpoints_Y = self.gridpoints_Y/max_length
			#self.gridpoints_Z = self.gridpoints_Z/max_length
			
			self.gridpoints = np.array(self.gridpoints)
			self.surface_normals = np.array(self.surface_normals)
			
			#norm vectors
			#self.gridpoints_normed = [g/g.abs() for g in self.gridpoints]
			#self.surface_normals_normed = [g/g.abs() for g in self.surface_normals]
			
			#getting coordinates of gridpoints
			self.gridpoints_X = np.array([i.x for i in self.gridpoints])
			self.gridpoints_Y = np.array([i.y for i in self.gridpoints])
			self.gridpoints_Z = np.array([i.z for i in self.gridpoints]) 

			#setting center to origo
			self.gridpoints_X = self.gridpoints_X - np.average(self.gridpoints_X)
			self.gridpoints_Y = self.gridpoints_Y - np.average(self.gridpoints_Y)
			self.gridpoints_Z = self.gridpoints_Z - np.average(self.gridpoints_Z)

			#getting coordinates of surface normals
			self.surface_normals_X = np.array([i.x for i in self.surface_normals])
			self.surface_normals_Y = np.array([i.y for i in self.surface_normals])
			self.surface_normals_Z = np.array([i.z for i in self.surface_normals])
			
			#getting coordinates of normed gridpoints
			#self.gridpoints_X_normed = np.array([i.x for i in self.gridpoints_normed])
			#self.gridpoints_Y_normed = np.array([i.y for i in self.gridpoints_normed])
			#self.gridpoints_Z_normed = np.array([i.z for i in self.gridpoints_normed]) 

			#setting center to origo for normed gridpoints
			#self.gridpoints_X_normed = self.gridpoints_X_normed - np.average(self.gridpoints_X_normed)
			#self.gridpoints_Y_normed = self.gridpoints_Y_normed - np.average(self.gridpoints_Y_normed)
			#self.gridpoints_Z_normed = self.gridpoints_Z_normed - np.average(self.gridpoints_Z_normed)

			#getting coordinates of surface normals
			#self.surface_normals_X_normed = np.array([i.x for i in self.surface_normals_normed])
			#self.surface_normals_Y_normed = np.array([i.y for i in self.surface_normals_normed])
			#self.surface_normals_Z_normed = np.array([i.z for i in self.surface_normals_normed])

			self.hp_theta = np.arctan2(self.gridpoints_Y, self.gridpoints_X)
			self.hp_phi = np.arctan2(self.gridpoints_Z, np.sqrt(np.power(self.gridpoints_X, 2)+np.power(self.gridpoints_Y, 2)))

			self.sn_theta = np.arctan2(self.surface_normals_Y, self.surface_normals_X)
			self.sn_phi = np.arctan2(self.surface_normals_Z, np.sqrt(np.power(self.surface_normals_X, 2)+np.power(self.surface_normals_Y, 2)))
			
			#calculating sigma angles
			self.sigma_s = np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_sun) + np.cos(self.hp_phi)*np.cos(self.phi_sun)*np.cos(self.hp_theta-self.theta_sun))
			self.sigma_n = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_sun) + np.cos(self.sn_phi)*np.cos(self.phi_sun)*np.cos(self.sn_theta-self.theta_sun))

			self.sigma_s_obs = np.arccos(np.sin(self.hp_phi)*np.sin(self.phi_obs) + np.cos(self.hp_phi)*np.cos(self.phi_obs)*np.cos(self.hp_theta-self.theta_obs))
			self.sigma_n_obs = np.arccos(np.sin(self.sn_phi)*np.sin(self.phi_obs) + np.cos(self.sn_phi)*np.cos(self.phi_obs)*np.cos(self.sn_theta-self.theta_obs))
		
		self.CreatedSurfaceGrid = True	
		print("\n\n >> Created \'", self.healpixmod, "\' surface grid for ", self.name, " object.", end='', sep='')
		print(" Calculated the surface normal coordinates and the sigma angle values.\n", end='', sep='')
			
	def spectral_emissivity(self, NN_WVL=81,	EPS_SPEC_WVLS0 =\
	np.array([1.0, 24.0, 70.0, 100.0, 160.0, 250.0, 350.0, 500.0, 1500.0]),\
	EPS_SPEC_VALS = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9,  0.9])):
		self.nn_wvl = NN_WVL
		self.eps_spec_wvls0 = EPS_SPEC_WVLS0
		self.eps_spec_vals = EPS_SPEC_VALS
		
		self.wvls = np.zeros(self.nn_wvl)
		self.nu = np.zeros(self.nn_wvl)
		self.dnu = np.zeros(self.nn_wvl)

		for i in range(self.nn_wvl):
			self.wvls[i] = 10**(log10(1500.0)/(self.nn_wvl)*i) #nn_wvl-1 kellene
			self.nu[i] = c_light/(self.wvls[i]*1.0e-6)

		self.dnu[0] = self.nu[0]

		for i in range(1,self.nn_wvl):
			self.dnu[i] = abs(self.nu[i] - self.nu[i-1])

		Epsilon_Lambda_Interpolate = interpolate.interp1d(self.eps_spec_wvls0, self.eps_spec_vals)
		self.eps_spec = np.zeros(self.nn_wvl)
		self.eps_spec = Epsilon_Lambda_Interpolate(self.wvls)
		
		print("\n\n >> Refreshed the spectral emissivity function of the ", self.name,\
		" object, interpolated for ", self.nn_wvl, " logarythmically spaced frequencies",\
		" and wavelengths. Don't forget to recalculate the surface temperature.\n", end='', sep='')
	
	def calculate_T(self):
		#-----------------------------------------------------------------------

		#EPSILON vs. T calculation: calculating the emissivity for each temperature

		self.T_array = np.linspace(1, 100)

		self.eps_bol = np.zeros(self.T_array.size)
		
		for i in tqdm(range(len(self.T_array)), desc="Calculating bolometric emissivity", bar_format='{l_bar:1}{bar:100}', ascii=' >#'):
			nominator = np.sum(self.eps_spec * B(self.nu, self.T_array[i]*np.ones(self.nu.size)) * self.dnu)
			denominator = np.sum(B(self.nu, self.T_array[i]*np.ones(self.nu.size)) * self.dnu)

			self.eps_bol[i] = nominator/denominator
			
		#-----------------------------------------------------------------------

		#Interpolation of the true surface temperature

		#INTERPOLATION OF epsilon*T**4
		self.et4 = (np.cos(self.sigma_n) > 0) * S_solar * (1.0-self.A_bol) * np.abs(np.cos(self.sigma_n)) / (self.r_helio**2 * sigma_sb * self.eta)# + (2.7**4.0) * (np.cos(sigma_n) <= 0).astype(float)
		g = interpolate.interp1d((self.eps_bol*np.power(self.T_array,4)), self.T_array, fill_value='extrapolate')
		self.Txy = g(self.et4)
		self.Txy[np.cos(self.sigma_n) <= 0] = self.Txy[np.cos(self.sigma_n) <= 0]*0.0 + 2.7
		#et4[np.cos(sigma_n) <= 0] = et4[np.cos(sigma_n) <= 0]*0.0 + 2.7**4.0
		self.eps_bol_xy = self.et4/np.power(self.Txy, 4)
		
		#-----------------------------------------------------------------------

		#SETTING ALL TEMPERATURES TO 2.7K ON THE "NIGHT SIDE"

		kkk = (np.cos(self.sigma_s) < 0.0)			#numpyarray with ones at the dark points
		kk = np.where(kkk == 1)[0]				#numpyarray with the i indices where kkk[i] == 1

		jj = np.nonzero((np.cos(self.sigma_s) > 0))

		self.et4[kk] = np.power(2.7,4.0)

		self.pVxy = self.Txy*0.0 + self.pV

		self.pVxy[kk] = 0.0
		
		print("\n\n >> Calculated surface temperature for ", self.name, " object.\n", end='', sep='')

	def project_to_focal_grid(self, N_PROJ_GRID=49):
		#-----------------------------------------------------------------------

		#projecting the body to the 'focal plane':
		#xx, yy, zz are the cartesian coordinates of the surface gridpoints
		#xx1 and xx2 are coordinates after two rotations to
		#align the y axis to (theta_obs, phi_obs)
		
		if self.healpixmod != "polyhedron":
			self.xx = self.a*np.cos(self.hp_theta)*np.cos(self.hp_phi)
			self.yy = self.b*np.sin(self.hp_theta)*np.cos(self.hp_phi)
			self.zz = self.c*np.sin(self.hp_phi)
		
		elif self.healpixmod == "polyhedron":
			self.xx = self.gridpoints_X
			self.yy = self.gridpoints_Y
			self.zz = self.gridpoints_Z

		#for each gridpoint, 1 if reached by radiation, 0 else
		self.lightside = (np.cos(self.sigma_s) > 0.0) * (np.cos(self.sigma_s_obs) > 0.0)

		#the points which are lightened
		self.zz_light = self.zz[np.nonzero(self.lightside)]
		self.xx_light = self.xx[np.nonzero(self.lightside)]
		self.yy_light = self.yy[np.nonzero(self.lightside)]

		#coordinates after the first rotation (along the z axis)
		self.xx1 = self.xx*np.cos(-self.theta_obs) - self.yy*np.sin(-self.theta_obs)
		self.yy1 = self.xx*np.sin(-self.theta_obs) + self.yy*np.cos(-self.theta_obs)
		self.zz1 = self.zz

		self.zz1_light = self.zz1[np.nonzero(self.lightside)]
		self.xx1_light = self.xx1[np.nonzero(self.lightside)]
		self.yy1_light = self.yy1[np.nonzero(self.lightside)]

		#coordinates after the second rotation (along the y axis)
		self.xx2 = self.xx1*np.cos(self.phi_obs) + self.zz1*np.sin(self.phi_obs)
		self.yy2 = self.yy1
		self.zz2 = -self.xx1*np.sin(self.phi_obs) + self.zz1*np.cos(self.phi_obs)

		self.zz2_light = self.zz2[np.nonzero(self.lightside)]
		self.xx2_light = self.xx2[np.nonzero(self.lightside)]
		self.yy2_light = self.yy2[np.nonzero(self.lightside)]

		#grid of the projected geometry

		self.nnn = N_PROJ_GRID		#number of sqruares in one row (column)
		gxyp = np.mgrid[-1:1+2/(self.nnn+1):2/(self.nnn+1), -1:1+2/(self.nnn+1):2/(self.nnn+1)]
			#(2, nnn+1, nnn+1)
		self.xxp = gxyp[0,:,:]
		self.zzp = gxyp[1,:,:]

		self.T2 = np.ones((self.nnn+2,self.nnn+2))*2.7					#temperature grid
		self.pV2 = np.zeros((self.nnn+2,self.nnn+2))						#???
		self.refl = np.zeros((self.nnn+2,self.nnn+2))						#???
		self.dx = 1/self.nnn		#2/nnn/2, dimensions of one square
		self.dz = 1/self.nnn

		#-----------------------------------------------------------------------

		#Projecting the surface temperature to the focal grid:
		
		for i in tqdm(range(self.nnn+1), desc="Projecting temperature to focal grid", bar_format='{l_bar:1}{bar:100}', ascii=' >#'):
			for j in range(self.nnn+1):
				iii = ((self.yy2 >= self.xxp[i,j]-self.dx) * (self.yy2 < self.xxp[i,j]+self.dx)\
				* (self.zz2 >= self.zzp[i,j]-self.dz) * (self.zz2 < self.zzp[i,j]+self.dz) \
				* (np.cos(self.sigma_s_obs) > 0.0) \
				* (np.cos(self.sigma_s) > 0.0))
					#* is equivalent with logical and, &&
					#iii is also a numpyarray filled with ones and zeros,
					#corresponding to where the given conditions are true/false
				ii = np.nonzero(iii)		
					#ii is a numpyarray with the indices where iii==1

				if len(ii[0]) != 0:
					self.T2[i,j] = np.amax(self.Txy[ii])
					#setting T to the maximum of temperatures
		self.ProjectedSurfaceGrid = True
		print("\n\n >> Projected the surface grid to the focal plane (", end='', sep='')
		print(self.name, ".T2 is the temperature distribution on the focal grid).", end='', sep='')
		print(" The focal grid has ", self.nnn, "×", self.nnn, " gridpoints.\n", end='', sep='')
			
					
	def mono_flux_density(self):
		#-----------------------------------------------------------------------

		#CALCULATION OF MONOCHROMATIC FLUX DENSITY
		#considering the spectral emissivity eps_spec(i) for each wavelength

		self.f_nu_t2 = np.zeros(self.nu.size)				#size: nn_wvl
		self.ii_t2 = np.nonzero((self.T2 > 0.0))
		self.ii_txy = np.nonzero((self.Txy > 0.0))

		if self.healpixmod != "cartesian" and self.healpixmod != "polyhedron":
			for i in tqdm(range(self.nn_wvl), desc="Calculating monochromatic flux density", bar_format='{l_bar:1}{bar:100}', ascii=' >#'):			#i runs from 0 to nn_wvl-1
				self.f_nu_t2[i] = 0.5*h_planck*self.nu[i]**3 / (c_light**2) * self.eps_spec[i] * (self.diam**2 / self.r_geo_km**2)\
				*np.sum( (np.exp(h_planck*self.nu[i]/k_b/self.T2[self.ii_t2]) - 1.0)**(-1) *4*self.dx*self.dz )
				
		elif self.healpixmod == "cartesian":
			for i in tqdm(range(self.nn_wvl), desc="Calculating monochromatic flux density", bar_format='{l_bar:1}{bar:100}', ascii=' >#'):			#i runs from 0 to nn_wvl-1
				self.f_nu_t2[i] = 0.5*h_planck*self.nu[i]**3 / (c_light**2) * self.eps_spec[i] * (self.diam**2 / self.r_geo_km**2)\
				*np.sum( (np.exp(h_planck*self.nu[i]/k_b/self.T2[self.ii_t2]) - 1.0)**(-1) *4*self.dx*self.dz ) 					
			
			self.f_nu = np.zeros(self.nu.size)
			#self.jj = np.nonzero(np.multiply((np.cos(self.sigma_s) > 0),(np.cos(self.sigma_n)>0)))
			self.jj = np.nonzero(np.cos(self.sigma_s) > 0)
			for i in tqdm(range(self.nn_wvl), desc="Calculating standard NEATM model for consistency check", bar_format='{l_bar:1}{bar:100}', ascii=' >#'):
				self.f_nu[i] = 0.5*h_planck*self.nu[i]**3 / (c_light**2) * self.eps * (self.diam**2/self.r_geo_km**2)\
				*np.sum( (np.exp(h_planck*self.nu[i]/k_b/self.Txy[self.jj])-1.0)**(-1) * np.power(np.cos(self.hp_phi[self.jj]), 2) * np.abs(np.cos(self.hp_theta[self.jj])) * self.dtheta * self.dphi )
			
			
		elif self.healpixmod == "polyhedron":
			self.lit_points = np.nonzero(np.multiply((np.cos(self.sigma_n_obs)>0),(np.cos(self.sigma_s_obs)>0))) #index of gridpoints which are enlightened
			self.proj_area = []
			self.n_obs = self.n_obs*(1/self.n_obs.abs())
			for v in self.surface_normals:
				self.proj_area.append(v**self.n_obs)
			self.proj_area = np.array(self.proj_area)
			#self.lit_points = np.nonzero((self.proj_area > 0))
			for i in tqdm(range(self.nn_wvl), desc="Calculating monochromatic flux density", bar_format='{l_bar:1}{bar:100}', ascii=' >#'):
				self.f_nu_t2[i] = 0.5*h_planck*self.nu[i]**3 / (c_light**2) * self.eps_spec[i] * (self.diam**2 / self.r_geo_km**2)\
				* np.sum((np.exp(h_planck*self.nu[i]/k_b/self.Txy[self.lit_points]) - 1.0)**(-1) * self.proj_area[self.lit_points])
				#np.sum(B(self.nu[i], self.Txy[self.lit_points])  * self.proj_area[self.lit_points])		
					
		print("\n\n >> Calculated monochromatic flux density (", self.name, ".f_nu_t2).\n", end='', sep='')
		self.calculated_f_mono = True
		
		
	def ib_flux_density(self, band_ib = np.array(['mips24', 'mips70', 'mips160', 'pacs70', 'pacs100', 'pacs160'\
		, 'spire250', 'spire350', 'spire500', 'alma6', 'alma7', 'alma8']),\
		wvl_cent = np.array([24.0, 71.42, 160., 70., 100., 160., 250., 350., 500., 1300., 1050., 870.])):
			
		#-----------------------------------------------------------------------

		#CALCULATION OF 'IN BAND' FLUX DENSITIES
		#for specific instruments/filters, with colour correction
		
		ip = interpolate.interp1d(self.wvls, self.eps_spec, fill_value='extrapolate')
		eps_cent = ip(wvl_cent)
		nu_cent = c_light/(wvl_cent*1.0e-6)

		self.f_nu_ib = np.zeros(band_ib.size)
		self.f_nu_mc = np.zeros(band_ib.size)
		self.ii_t2 = np.nonzero((self.T2 > 21.0))
		self.ii_txy = np.nonzero((self.Txy > 21.0))

		if self.healpixmod != "cartesian" and self.healpixmod != "polyhedron":
			print("...")

		elif self.healpixmod == "cartesian":
			for i in range(band_ib.size):
				self.f_nu_ib[i] = 0.5 * h_planck * np.power(nu_cent[i],3) / (c_light**2) * eps_cent[i] * (self.diam**2 / self.r_geo_km**2)\
				*np.sum( cc(self.T2[self.ii_t2],band_ib[i]) * (np.exp(h_planck*nu_cent[i]/k_b/self.T2[self.ii_t2]) - 1.0)**(-1) *4*self.dx*self.dz )

				self.f_nu_mc[i] = 0.5 * h_planck * np.power(nu_cent[i],3) / (c_light**2) * eps_cent[i] * (self.diam**2 / self.r_geo_km**2)\
				*np.sum( (np.exp(h_planck*nu_cent[i]/k_b/self.T2[self.ii_t2])-1)**(-1) ) *self.dx*self.dz*4.0
		
		elif self.healpixmod == "polyhedron": #!!! itt még javítani kell
			self.lit_points = np.nonzero((np.cos(self.sigma_n_obs)>0)) #index of gridpoints which are enlightened
			self.proj_area = []
			for v in self.surface_normals:
				self.proj_area.append(v**self.n_obs)
			self.proj_area = np.array(self.proj_area)*self.diam
			for i in range(self.nn_wvl):
				self.f_nu_ib[i] = (1 / self.r_geo_km**2) * np.sum(B(self.nu[i], self.Txy[self.lit_points])  * self.proj_area[self.lit_points])
				
		print("\n\n >> Calculated in-band flux density (", self.name, ".f_nu_ib) and the corresponding monochromatic flux density (", self.name, ".f_nu_mc)", end='', sep='')
		print(" for the given instruments and center wavelengths.\n", end='', sep='')
		self.calculated_f_ib = True
		
	def plot_theta_phi(self):
		if not self.CreatedSurfaceGrid:
			print("\n\n >> No surface grid created on the ", self.name, " object. Use the ", self.name,\
			".creategrid() function to make one.", end='', sep='')
			
		else:
			plt.rcParams["figure.figsize"] = [6.0, 6.0]
			plt.rcParams["figure.autolayout"] = True

			plt.title("Surface grid", fontweight='bold')
			plt.ylabel(r"$\mathbf{hp\phi}$", fontweight='bold')
			plt.xlabel(r"$\mathbf{hp\theta}$", fontweight='bold')
			plt.scatter(self.hp_theta, self.hp_phi, color='black', s=1)
			plt.show()
			
		
	def plot_T(self):
		### PLOT T2 ###
		
		if not self.CreatedSurfaceGrid:
			print("\n\n >> No surface grid created on the ", self.name, " object. Use the ", self.name,\
			".creategrid() function to make one.", end='', sep='')
			
		if not self.ProjectedSurfaceGrid:
			print("\n\n >> Surface grid of the ", self.name, " object was not projected to the focal grid.\
			 Use the ", self.name,"project_to_focal_grid(N_PROJ_GRID) function to do so.", end='', sep='')
		
		else:
			plt.rcParams["figure.figsize"] = [6.0, 6.0]
			plt.rcParams["figure.autolayout"] = True

			#plt.title(r"$a=1, b=0.8, c=0.8, \theta_{\odot}=0, \phi_{\odot}=10, \theta_{obs}=35, \phi_{obs}=30$")
			plt.ylabel(r"$\mathbf{y}$", fontweight='bold')
			plt.xlabel(r"$\mathbf{x}$", fontweight='bold')
			#plt.scatter(xxp, T2, color=clr, s=linewidth)
			plt.pcolor(self.T2,cmap='hot')

			plt.show()
			
	def plot_f_mono(self):
		if not self.CreatedSurfaceGrid:
			print("\n\n >> No surface grid created on the ", self.name, " object. Use the ", self.name,\
			".creategrid() function to make one.", end='', sep='')
			
		if not self.ProjectedSurfaceGrid:
			print("\n\n >> Surface grid of the ", self.name, " object was not projected to the focal grid.\
			 Use the ", self.name,"project_to_focal_grid(N_PROJ_GRID) function to do so.", end='', sep='')
			
		if not self.calculated_f_mono:
			print("\n\n >> Monochromatic flux density of the ", self.name, " object was not calculated.\
			 Use the ", self.name,"mono_flux_density function to do so.", end='', sep='')
		
		else:
			plt.rcParams["figure.figsize"] = [6.0, 6.0]
			plt.rcParams["figure.autolayout"] = True

			plt.title("Monochromatic flux density")
			plt.ylabel(r"$\mathbf{f_{\lambda}}$ [mJy]", fontweight='bold')
			plt.xlabel(r"$\lambda$", fontweight='bold')
			plt.grid()
			ax = plt.gca()
			ax.set_xscale('log')
			ax.set_yscale('log')
			ax.set_ylim(0.01, 10000)
			ax.scatter(self.wvls, self.f_nu_t2*1e29, color="red", s=5, marker='^')

			plt.show()
		
						###END OF MODULE###
