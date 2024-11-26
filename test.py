import matplotlib.pyplot as plt
import numpy as np
import sys

import rinex as rn
import gnss as gn
from gnss import rSigRnx, time2str, uGNSS
from rtk import rtkpos

xyz_ref = [-3962108.7007, 3381309.5532, 3668678.6648]
pos_ref = gn.ecef2pos(xyz_ref)

sigs = [rSigRnx("GC1C"), rSigRnx("GC5Q"),
        rSigRnx("EC1C"), rSigRnx("EC5Q"),
        rSigRnx("GL1C"), rSigRnx("GL5Q"),
        rSigRnx("EL1C"), rSigRnx("EL5Q"),
        rSigRnx("GS1C"), rSigRnx("GS5Q"),
        rSigRnx("ES1C"), rSigRnx("ES5Q")]

sigsb = [rSigRnx("GC1C"), rSigRnx("GC5X"),
         rSigRnx("EC1X"), rSigRnx("EC5X"),
         rSigRnx("GL1C"), rSigRnx("GL5X"),
         rSigRnx("EL1X"), rSigRnx("EL5X"),
         rSigRnx("GS1C"), rSigRnx("GS5X"),
         rSigRnx("ES1X"), rSigRnx("ES5X")]
         
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

nep = 10  # 3 minutes
t = np.zeros(nep)
enu = np.zeros((nep, 3))
smode = np.zeros(nep, dtype=int)

# Arrays to store ionospheric delays and phase combinations
p1 = np.zeros((nep, uGNSS.MAXSAT))
p2 = np.zeros((nep, uGNSS.MAXSAT))
c1 = np.zeros((nep, uGNSS.MAXSAT))
c2 = np.zeros((nep, uGNSS.MAXSAT))
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
    p1[ne, :] = nav.p1
    c1[ne, :] = nav.c1
    p2[ne, :] = nav.p2
    c2[ne, :] = nav.c2    
    
    # Carrier smoothing for pseudorange
    if ne > 0:
        smoothed_piono[ne, :] = weight * piono[ne, :] + (1 - weight) * (smoothed_piono[ne - 1, :] + ciono[ne, :] - ciono[ne - 1, :])
    else:
        smoothed_piono[ne, :] = piono[ne, :]
    

dec.fobs.close()
decb.fobs.close()

def plt_enu(t, enu):
    plt.figure(figsize=(5,4))
    plt.plot(t, enu)
    plt.ylabel('pos err[m]')
    plt.xlabel('time[s]')
    plt.legend(['east', 'north', 'up'])
    plt.grid()
    plt.xlim([0, nep])
    plt.show()

plt_enu(t, enu)

# Function to plot measurements for each satellite separately
def plt_meas(t, measurement, title, exclude_sat_ids=[]):
    """
    Function to plot measurements for each satellite separately, with optional exclusion of specific satellites.

    Parameters:
    - t: Time array
    - measurement: Measurement data (time x satellites)
    - title: Title for the plot
    - exclude_sat_ids: List of satellite IDs to exclude from plotting
    """
    # Create a figure with 2 subplots (2 rows, 1 column)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12), sharex=True)  # 2 rows, 1 column
    
    # Plot GPS measurements in the first subplot (ax1)
    for sat_id in range(uGNSS.GPSMAX):
        if sat_id not in exclude_sat_ids and np.any(measurement[:, sat_id]):  # Exclude specific satellites
            ax1.plot(t, measurement[:, sat_id], label=f'{sat_id+1}')
    ax1.set_ylabel('GPS (m)')
    ax1.grid()
    ax1.legend(loc='upper right', bbox_to_anchor=(1.05, 1))

    # Plot Galileo measurements in the second subplot (ax2)
    for sat_id in range(uGNSS.GPSMAX, uGNSS.GPSMAX + uGNSS.GALMAX):
        if sat_id not in exclude_sat_ids and np.any(measurement[:, sat_id]):  # Exclude specific satellites
            ax2.plot(t, measurement[:, sat_id], label=f'{sat_id-uGNSS.GPSMAX+1}')
    ax2.set_ylabel('Galileo (m)')
    ax2.set_xlabel('Time (s)')
    ax2.grid()
    ax2.legend(loc='upper right', bbox_to_anchor=(1.05, 1))
    
    # Set the title for the figure
    fig.suptitle(title)

    # Adjust layout to prevent overlap between subplots
    plt.tight_layout()
    ax1.set_xlim([0, nep])
    ax2.set_xlim([0, nep])

    plt.show()


# Plot measurments
plt_meas(t, iono, "Estimated Iono")
plt_meas(t, piono, "Iono from Phase Geometry Free Comb")
plt_meas(t, ciono, "Iono from Code Geometry Free Comb")
plt_meas(t, smoothed_piono, "Smoothed Iono")
plt_meas(t, p1, "1st Phase")
plt_meas(t, p2, "2nd Phase")
plt_meas(t, c1, "1st Code")
plt_meas(t, c2, "2nd Code")
