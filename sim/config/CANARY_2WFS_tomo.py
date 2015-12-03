"""
Config File for python tomographic AO simulation.

This configuration file contains a python dictionary in which all simulation parameters are saved. The main dictionary ``simConfig``, is split into several sections corresponding to different parts of the simulation, each of which is itself another python dictionary

This file will be parsed and missing parameters will trigger warnings, either defaulting to a reasonable value or exiting the simulation. Python syntax errors in this file can stop the simulation from running, ensure that every pararmeter finishes with a comma, including the sub-dictionaries.
"""
import numpy

simConfiguration = {

"Sim":{
    "simName"       :  "CANARY_2WFS_tomo",
    "logfile"       :   "CANARY_2WFS_tomo.log",
    "pupilSize"     :   64,
    "nGS"           :   2,
    "nDM"           :   0,
    "nSci"          :   0,
    "nIters"        :   20000,
    "loopTime"      :   1/400.0,
    "reconstructor" :   "MVM",

    "verbosity"     :   2,

    "saveSlopes"    :   True,
    },

"Atmosphere":{
    "scrnNo"        :   1,
    "scrnHeights"   :   numpy.array([12376, 5000, 10000, 15000]),
    "scrnStrengths" :   numpy.array([0.5, 0.3, 0.1, 0.1]),
    "windDirs"      :   numpy.array([10, 45, 90, 135]),
    "windSpeeds"    :   numpy.array([10, 10, 15, 20]),
    "wholeScrnSize" :   4096,
    "r0"            :   0.16,
    "randomScrns"   :   True,
    "subHarmonics"  :   True,
    },

"Telescope":{
   "telDiam"        :   4.2,  # Metres
   "obsDiam"        :   0, # Central Obscuration
   "mask"           :   "circle",
    },

"WFS":{
    "GSPosition"    :   [(0,5),(0,-5)],
    "nxSubaps"      :   [7]*2,
    "pxlsPerSubap"  :   [16]*2,
    "subapFOV"      :   [3.]*2,
    "fftOversamp"   :   [3]*2,
    "wavelength"    :   [600e-9]*2,
    "centMethod"    :   ["brightestPxl"]*2,
    "centThreshold" :   [0.1]*2,

    },

"LGS":{

    },

"DM":{

    },

"Science":{

    }
}
