#IMPORTING MODULS
from math import *
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy import interpolate, integrate
from neatm_ell_4gh_v8 import neatm
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

haumea = neatm("parameters.dat")
haumea.help()
haumea.create_grid()
haumea.calculate_T()
haumea.project_to_focal_grid(49)
haumea.mono_flux_density()
haumea.plot_T()
haumea.plot_f_mono()
