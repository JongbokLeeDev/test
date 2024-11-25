import matplotlib.pyplot as plt
import numpy as np
import sys

import rinex as rn
import gnss as gn
from gnss import rSigRnx, time2str, uGNSS
from rtk import rtkpos

xyz_ref = [-3962108.7007, 3381309.5532, 3668678.6648]
pos_ref = gn.ecef2pos(xyz_ref)

sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"),
        rSigRnx("EC1C"), rSigRnx("EC5Q"),
        rSigRnx("JC1C"), rSigRnx("JC2L"),
        rSigRnx("GL1C"), rSigRnx("GL2W"),
        rSigRnx("EL1C"), rSigRnx("EL5Q"),
        rSigRnx("JL1C"), rSigRnx("JL2L"),
        rSigRnx("GS1C"), rSigRnx("GS2W"),
        rSigRnx("ES1C"), rSigRnx("ES5Q"),
        rSigRnx("JS1C"), rSigRnx("JS2L")]

sigsb = [rSigRnx("GC1C"), rSigRnx("GC2W"),
         rSigRnx("EC1X"), rSigRnx("EC5X"),
         rSigRnx("JC1X"), rSigRnx("JC2X"),
         rSigRnx("GL1C"), rSigRnx("GL2W"),
         rSigRnx("EL1X"), rSigRnx("EL5X"),
         rSigRnx("JL1X"), rSigRnx("JL2X"),
         rSigRnx("GS1C"), rSigRnx("GS2W"),
         rSigRnx("ES1X"), rSigRnx("ES5X"),
         rSigRnx("JS1X"), rSigRnx("JS2X")]
         
bdir = 'data/'
navfile = bdir+'SEPT238A.23P'
obsfile = bdir+'SEPT238A.23O'

# rover
dec = rn.rnxdec()
dec.setSignals(sigs)
nav = gn.Nav()
dec.decode_nav(navfile, nav)
dec.decode_obsh(obsfile)
dec.autoSubstituteSignals()

# base station
basefile = bdir+'3034238A.23O'
nav.rb = [-3959400.6443, 3385704.4948, 3667523.1275]  # GSI 3034 fujisawa
decb = rn.rnxdec()
decb.setSignals(sigsb)
decb.decode_obsh(basefile)
decb.autoSubstituteSignals()

rtk = rtkpos(nav, dec.pos, 'test_rtk.log')
rr = dec.pos

nep = 3 * 60  # 3 minutes
t = np.zeros(nep)
enu = np.zeros((nep, 3))
smode = np.zeros(nep, dtype=int)

# Arrays to store ionospheric delays and phase combinations
iono = np.zeros((nep, uGNSS.MAXSAT))
piono = np.zeros((nep, uGNSS.MAXSAT))
ciono = np.zeros((nep, uGNSS.MAXSAT))
smoothed_piono = np.zeros((nep, uGNSS.MAXSAT))

# Weight for smoothing
weight = 0.1

for ne in range(nep):
    obs, obsb = rn.sync_obs(dec, decb)
    if ne == 0:
        t0 = nav.t = obs.t
    rtk.process(obs, obsb=obsb)
    t[ne] = gn.timediff(nav.t, t0)
    sol = nav.xa[0:3] if nav.smode == 4 else nav.x[0:3]
    enu[ne, :] = gn.ecef2enu(pos_ref, sol-xyz_ref)
    smode[ne] = nav.smode
    # Log to standard output
    sys.stdout.write('\r {} ENU {:7.4f} {:7.4f} {:7.4f}, 2D {:6.4f}, mode {:1d}'
                     .format(time2str(obs.t),
                             enu[ne, 0], enu[ne, 1], enu[ne, 2],
                             np.sqrt(enu[ne, 0]**2+enu[ne, 1]**2),
                             smode[ne]))
    
    # Store ionospheric delays
    iono[ne, :] = nav.xa[7:7 + uGNSS.MAXSAT] if nav.smode == 4 else nav.x[7:7 + uGNSS.MAXSAT]
    piono[ne, :] = nav.piono
    ciono[ne, :] = nav.ciono
    
    # Carrier smoothing for pseudorange
    if ne > 0:
        smoothed_piono[ne, :] = weight * piono[ne, :] + (1 - weight) * (smoothed_piono[ne - 1, :] + ciono[ne, :] - ciono[ne - 1, :])
    else:
        smoothed_piono[ne, :] = piono[ne, :]
    

dec.fobs.close()
decb.fobs.close()

def plt_enu(t, enu, dmax=0.1):
    plt.figure(figsize=(5,4))
    plt.plot(t, enu)
    plt.ylabel('pos err[m]')
    plt.xlabel('time[s]')
    plt.legend(['east', 'north', 'up'])
    plt.grid()
    plt.axis([0, nep, -dmax, dmax])
    plt.show()

plt_enu(t, enu)

# Function to plot ionospheric delays for each satellite separately
def plt_iono(t, iono, dmax=1.0, label='Ionospheric Delays'):
    plt.figure(figsize=(10, 6))
    
    # Loop over each satellite and plot its ionospheric delay
    for sat_id in range(uGNSS.MAXSAT):
        if np.any(iono[:, sat_id]):  # Only plot if there is data for this satellite
            plt.plot(t, iono[:, sat_id], label=f'Satellite {sat_id+1} {label}')
    
    plt.ylabel('Ionospheric Delays (m)')
    plt.xlabel('Time (s)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
    plt.grid()
    plt.axis([0, nep, -dmax, dmax])
    plt.show()

# Plot ionospheric delays and smoothed ionospheric delays
plt_iono(t, iono)
plt_iono(t, smoothed_piono)
