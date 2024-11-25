"""
module for standard PPP positioning
"""

import numpy as np

from ephemeris import satposs
from gnss import sat2id, sat2prn, rSigRnx, uTYP, uGNSS, rCST
from gnss import uTropoModel, ecef2pos, tropmodel, geodist, satazel
from gnss import time2str, timediff, gpst2utc, tropmapf, uIonoModel
from mlambda import mlambda

# format definition for logging
fmt_ztd = "{}         ztd      ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f}\n"
fmt_ion = "{} {}-{} ion {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f}\n"
fmt_res = "{} {}-{} res {} ({:3d}) {:10.3f} sig_i {:10.3f} sig_j {:10.3f}\n"
fmt_amb = "{} {}-{} amb {} ({:3d},{:3d}) {:10.3f} {:10.3f} {:10.3f} " + \
    "{:10.3f} {:10.3f} {:10.3f}\n"


class pppos():
    """ class for PPP processing """

    nav = None

    def __init__(self, nav, pos0=np.zeros(3),
                 logfile=None, trop_opt=1, iono_opt=1, phw_opt=1):
        """ initialize variables for PPP """

        self.nav = nav

        # Number of frequencies (actually signals!)
        #
        self.nav.ephopt = 2  # SSR-APC

        # Select tropospheric model
        #
        self.nav.trpModel = uTropoModel.SAAST

        # Select iono model
        #
        self.nav.ionoModel = uIonoModel.KLOBUCHAR

        # 0: use trop-model, 1: estimate, 2: use cssr correction
        self.nav.trop_opt = trop_opt

        # 0: use iono-model, 1: estimate, 2: use cssr correction
        self.nav.iono_opt = iono_opt

        # 0: none, 1: full model, 2: local/regional model
        self.nav.phw_opt = phw_opt

        # carrier smoothing
        self.nav.csmooth = False

        # Position (+ optional velocity), zenith tropo delay and
        # slant ionospheric delay states
        #
        self.nav.ntrop = (1 if self.nav.trop_opt == 1 else 0)
        self.nav.niono = (uGNSS.MAXSAT if self.nav.iono_opt == 1 else 0)

        self.nav.na = (3 if self.nav.pmode == 0 else 6)
        self.nav.nq = (3 if self.nav.pmode == 0 else 6)

        self.nav.na += self.nav.ntrop + self.nav.niono
        self.nav.nq += self.nav.ntrop + self.nav.niono

        # State vector dimensions (including slant iono delay and ambiguities)
        #
        self.nav.nx = self.nav.na+uGNSS.MAXSAT*self.nav.nf

        self.nav.x = np.zeros(self.nav.nx)
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))

        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))

        self.nav.phw = np.zeros(uGNSS.MAXSAT)
        self.nav.el = np.zeros(uGNSS.MAXSAT)
        
        self.nav.ciono = np.zeros(uGNSS.MAXSAT) # iono from code geometry free linear combination
        self.nav.piono = np.zeros(uGNSS.MAXSAT) # iono from phase geometry free linear combination

        # Parameters for PPP
        #
        # Observation noise parameters
        #
        self.nav.eratio = np.ones(self.nav.nf)*100  # [-] factor
        self.nav.err = [0, 0.000, 0.003]       # [m] sigma

        # Initial sigma for state covariance
        #
        self.nav.sig_p0 = 100.0   # [m]
        self.nav.sig_v0 = 1.0     # [m/s]
        self.nav.sig_ztd0 = 0.1  # [m]
        self.nav.sig_ion0 = 10.0  # [m]
        self.nav.sig_n0 = 30.0    # [cyc]

        # Process noise sigma
        #
        if self.nav.pmode == 0:
            self.nav.sig_qp = 100.0/np.sqrt(1)     # [m/sqrt(s)]
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)      # [m/sqrt(s)]
            self.nav.sig_qv = 1.0/np.sqrt(1)       # [m/s/sqrt(s)]
        self.nav.sig_qztd = 0.05/np.sqrt(3600)     # [m/sqrt(s)]
        self.nav.sig_qion = 10.0/np.sqrt(1)        # [m/s/sqrt(s)]

        # Processing options
        #
        self.nav.thresar = 3.0  # AR acceptance threshold
        # 0:float-ppp,1:continuous,2:instantaneous,3:fix-and-hold
        self.nav.armode = 0
        self.nav.elmaskar = np.deg2rad(20.0)  # elevation mask for AR
        self.nav.elmin = np.deg2rad(10.0)

        # Initial state vector
        #
        self.nav.x[0:3] = pos0
        if self.nav.pmode >= 1:  # kinematic
            self.nav.x[3:6] = 0.0  # velocity

        # Diagonal elements of covariance matrix
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True

        dP[0:3] = self.nav.sig_p0**2
        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            dP[3:6] = self.nav.sig_v0**2

        # Tropo delay
        if self.nav.trop_opt == 1:  # trop is estimated
            if self.nav.pmode >= 1:  # kinematic
                dP[6] = self.nav.sig_ztd0**2
            else:
                dP[3] = self.nav.sig_ztd0**2

        # Process noise
        #
        self.nav.q = np.zeros(self.nav.nq)
        self.nav.q[0:3] = self.nav.sig_qp**2

        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            self.nav.q[3:6] = self.nav.sig_qv**2

        if self.nav.trop_opt == 1:  # trop is estimated
            # Tropo delay
            if self.nav.pmode >= 1:  # kinematic
                self.nav.q[6] = self.nav.sig_qztd**2
            else:
                self.nav.q[3] = self.nav.sig_qztd**2

        if self.nav.iono_opt == 1:  # iono is estimated
            # Iono delay
            if self.nav.pmode >= 1:  # kinematic
                self.nav.q[7:7+uGNSS.MAXSAT] = self.nav.sig_qion**2
            else:
                self.nav.q[4:4+uGNSS.MAXSAT] = self.nav.sig_qion**2

        # Logging level
        #
        self.monlevel = 0
        self.nav.fout = None
        if logfile is None:
            self.nav.monlevel = 0
        else:
            self.nav.fout = open(logfile, 'w')

    def valpos(self, v, R, thres=4.0):
        """ post-fit residual test """
        nv = len(v)
        fact = thres**2
        for i in range(nv):
            if v[i]**2 <= fact*R[i, i]:
                continue
            if self.nav.monlevel > 1:
                txt = "{:3d} is large: {:8.4f} ({:8.4f})".format(
                    i, v[i], R[i, i])
                if self.nav.fout is None:
                    print(txt)
                else:
                    self.nav.fout.write(txt+"\n")
        return True

    def initx(self, x0, v0, i):
        """ initialize x and P for index i """
        self.nav.x[i] = x0
        for j in range(self.nav.nx):
            self.nav.P[j, i] = self.nav.P[i, j] = v0 if i == j else 0

    def IB(self, s, f, na=3):
        """ return index of phase ambguity """
        idx = na+uGNSS.MAXSAT*f+s-1
        return idx

    def II(self, s, na):
        """ return index of slant ionospheric delay estimate """
        return na-uGNSS.MAXSAT+s-1

    def IT(self, na):
        """ return index of zenith tropospheric delay estimate """
        return na-uGNSS.MAXSAT-1

    def varerr(self, nav, el, f):
        """ variation of measurement """
        s_el = max(np.sin(el), 0.1*rCST.D2R)
        fact = nav.eratio[f-nav.nf] if f >= nav.nf else 1
        a = fact*nav.err[1]
        b = fact*nav.err[2]
        return (a**2+(b/s_el)**2)

    def sysidx(self, satlist, sys_ref):
        """ return index of satellites with sys=sys_ref """
        idx = []
        for k, sat in enumerate(satlist):
            sys, _ = sat2prn(sat)
            if sys == sys_ref:
                idx.append(k)
        return idx

    def udstate(self, obs):
        """ time propagation of states and initialize """

        tt = timediff(obs.t, self.nav.t)

        ns = len(obs.sat)
        sys = []
        sat = obs.sat
        for sat_i in obs.sat:
            sys_i, _ = sat2prn(sat_i)
            sys.append(sys_i)

        # pos,vel,ztd,ion,amb
        #
        nx = self.nav.nx
        Phi = np.eye(nx)
        # if self.nav.niono > 0:
        #    ni = self.nav.na-uGNSS.MAXSAT
        #    Phi[ni:self.nav.na, ni:self.nav.na] = np.zeros(
        #        (uGNSS.MAXSAT, uGNSS.MAXSAT))
        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6]*tt
            Phi[0:3, 3:6] = np.eye(3)*tt
        self.nav.P[0:nx, 0:nx] = Phi@self.nav.P[0:nx, 0:nx]@Phi.T

        # Process noise
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        dP[0:self.nav.nq] += self.nav.q[0:self.nav.nq]*tt

        # Update Kalman filter state elements
        #
        for f in range(self.nav.nf):

            # Reset phase-ambiguity if instantaneous AR
            # or expire obs outage counter
            #
            for i in range(uGNSS.MAXSAT):

                sat_ = i+1
                sys_i, _ = sat2prn(sat_)

                self.nav.outc[i, f] += 1
                reset = (self.nav.outc[i, f] >
                         self.nav.maxout or np.any(self.nav.edt[i, :] > 0))
                if sys_i not in obs.sig.keys():
                    continue

                # Reset ambiguity estimate
                #
                j = self.IB(sat_, f, self.nav.na)
                if reset and self.nav.x[j] != 0.0:
                    self.initx(0.0, 0.0, j)
                    self.nav.outc[i, f] = 0

                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - reset ambiguity  {}\n"
                            .format(time2str(obs.t), sat2id(sat_),
                                    obs.sig[sys_i][uTYP.L][f]))

                if self.nav.niono > 0:
                    # Reset slant ionospheric delay estimate
                    #
                    j = self.II(sat_, self.nav.na)
                    if reset and self.nav.x[j] != 0.0:
                        self.initx(0.0, 0.0, j)

                        if self.nav.monlevel > 0:
                            self.nav.fout.write("{}  {} - reset ionosphere\n"
                                                .format(time2str(obs.t),
                                                        sat2id(sat_)))

            # Ambiguity
            #
            bias = np.zeros(ns)
            ion = np.zeros(ns)
            f1 = 0

            """
            offset = 0
            na = 0
            """
            for i in range(ns):

                # Do not initialize invalid observations
                #
                if np.any(self.nav.edt[sat[i]-1, :] > 0):
                    continue

                if self.nav.nf > 1 and self.nav.niono > 0:
                    # Get dual-frequency pseudoranges for this constellation
                    #
                    sig1 = obs.sig[sys[i]][uTYP.C][0]
                    sig2 = obs.sig[sys[i]][uTYP.C][1]

                    pr1 = obs.P[i, 0]
                    pr2 = obs.P[i, 1]

                    # Skip zero observations
                    #
                    if pr1 == 0.0 or pr2 == 0.0:
                        continue

                    if sys[i] == uGNSS.GLO:
                        if sat[i] not in self.nav.glo_ch:
                            print("glonass channed not found: {:d}"
                                  .format(sat[i]))
                            continue
                        f1 = sig1.frequency(self.nav.glo_ch[sat[i]])
                        f2 = sig2.frequency(self.nav.glo_ch[sat[i]])
                    else:
                        f1 = sig1.frequency()
                        f2 = sig2.frequency()

                    # Get iono delay at frequency of first signal
                    #
                    ion[i] = (pr1-pr2)/(1.0-(f1/f2)**2)

                # Get pseudorange and carrier-phase observation of signal f
                #
                sig = obs.sig[sys[i]][uTYP.L][f]

                if sys[i] == uGNSS.GLO:
                    fi = sig.frequency(self.nav.glo_ch[sat[i]])
                else:
                    fi = sig.frequency()

                lam = rCST.CLIGHT/fi

                cp = obs.L[i, f]
                pr = obs.P[i, f]
                if cp == 0.0 or pr == 0.0 or lam is None:
                    continue

                bias[i] = cp - pr/lam + 2.0*ion[i]/lam*(f1/fi)**2

                """
                amb = nav.x[IB(sat[i], f, nav.na)]
                if amb != 0.0:
                    offset += bias[i] - amb
                    na += 1
                """
            """
            # Adjust phase-code coherency
            #
            if na > 0:
                db = offset/na
                for i in range(uGNSS.MAXSAT):
                    if nav.x[IB(i+1, f, nav.na)] != 0.0:
                        nav.x[IB(i+1, f, nav.na)] += db
            """

            # Initialize ambiguity
            #
            for i in range(ns):

                sys_i, _ = sat2prn(sat[i])

                j = self.IB(sat[i], f, self.nav.na)
                if bias[i] != 0.0 and self.nav.x[j] == 0.0:

                    self.initx(bias[i], self.nav.sig_n0**2, j)

                    if self.nav.monlevel > 0:
                        sig = obs.sig[sys_i][uTYP.L][f]
                        self.nav.fout.write(
                            "{}  {} - init  ambiguity  {} {:12.3f}\n"
                            .format(time2str(obs.t), sat2id(sat[i]),
                                    sig, bias[i]))

                if self.nav.niono > 0:
                    j = self.II(sat[i], self.nav.na)
                    if ion[i] != 0 and self.nav.x[j] == 0.0:

                        self.initx(ion[i], self.nav.sig_ion0**2, j)

                        if self.nav.monlevel > 0:
                            self.nav.fout.write(
                                "{}  {} - init  ionosphere      {:12.3f}\n"
                                .format(time2str(obs.t), sat2id(sat[i]),
                                        ion[i]))

        return 0

    def zdres(self, obs, cs, bsx, rs, vs, dts, rr, rtype=1):
        """ non-differential residual """

        _c = rCST.CLIGHT
        ns2m = _c*1e-9

        nf = self.nav.nf
        n = len(obs.P)
        y = np.zeros((n, nf*2))
        el = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = rr.copy()

        # Geodetic position
        #
        pos = ecef2pos(rr_)

        # Zenith tropospheric dry and wet delays at user position
        #
        trop_hs, trop_wet, _ = tropmodel(obs.t, pos,
                                         model=self.nav.trpModel)

        cpc = np.zeros((n, nf))
        prc = np.zeros((n, nf))

        for i in range(n):

            sat = obs.sat[i]
            sys, _ = sat2prn(sat)

            # Skip edited observations
            #
            if np.any(self.nav.edt[sat-1, :] > 0):
                continue

            # Pseudorange, carrier-phase and C/N0 signals
            #
            sigsPR = obs.sig[sys][uTYP.C]
            sigsCP = obs.sig[sys][uTYP.L]

            # Wavelength
            #
            if sys == uGNSS.GLO:
                lam = np.array([s.wavelength(self.nav.glo_ch[sat])
                                for s in sigsCP])
                frq = np.array([s.frequency(self.nav.glo_ch[sat])
                               for s in sigsCP])
            else:
                lam = np.array([s.wavelength() for s in sigsCP])
                frq = np.array([s.frequency() for s in sigsCP])

            # Geometric distance corrected for Earth rotation
            # during flight time
            #
            r, e[i, :] = geodist(rs[i, :], rr_)
            _, el[i] = satazel(pos, e[i, :])
            if el[i] < self.nav.elmin:
                continue

            # Tropospheric delay mapping functions
            #
            mapfh, mapfw = tropmapf(obs.t, pos, el[i],
                                    model=self.nav.trpModel)

            # Tropospheric delay
            #
            trop = mapfh*trop_hs + mapfw*trop_wet

            # Ionospheric delay
            iono = np.zeros(nf)

            # Range correction
            #
            prc[i, :] = trop + iono
            cpc[i, :] = trop - iono

            r -= _c*dts[i]

            for f in range(nf):
                y[i, f] = obs.L[i, f]*lam[f]-(r+cpc[i, f])
                y[i, f+nf] = obs.P[i, f]-(r+prc[i, f])

        return y, e, el

    def sdres(self, obs, x, y, e, sat, el):
        """
        SD phase/code residuals

        Parameters
        ----------

        obs : Obs()
            Data structure with observations
        x   :
            State vector elements
        y   :
            Un-differenced corrected observations
        e   :
            Line-of-sight vectors
        sat : np.array of int
            List of satellites
        el  : np.array of float values
            Elevation angles

        Returns
        -------
        v   : np.array of float values
            Residuals of single-difference measurements
        H   : np.array of float values
            Jacobian matrix with partial derivatives of state variables
        R   : np.array of float values
            Covariance matrix of single-difference measurements
        """

        nf = self.nav.nf  # number of frequencies (or signals)
        ns = len(el)  # number of satellites
        nc = len(obs.sig.keys())  # number of constellations

        mode = 1 if len(y) == ns else 0  # 0:DD,1:SD

        nb = np.zeros(2*nc*nf, dtype=int)

        Ri = np.zeros(ns*nf*2)
        Rj = np.zeros(ns*nf*2)

        nv = 0
        b = 0

        H = np.zeros((ns*nf*2, self.nav.nx))
        v = np.zeros(ns*nf*2)

        # Geodetic position
        #
        pos = ecef2pos(x[0:3])

        # Loop over constellations
        #
        for sys in obs.sig.keys():

            # Loop over twice the number of frequencies
            #   first for all carrier-phase observations
            #   second all pseudorange observations
            #
            for f in range(0, nf*2):
                # Select satellites from one constellation only
                #
                idx = self.sysidx(sat, sys)

                if len(idx) == 0:
                    continue

                # Select reference satellite with highest elevation
                #
                i = idx[np.argmax(el[idx])]

                # Loop over satellites
                #
                for j in idx:

                    # Slant ionospheric delay reference frequency
                    #
                    if sys == uGNSS.GLO:
                        freq0 = obs.sig[sys][uTYP.L][0].frequency(0)
                    else:
                        freq0 = obs.sig[sys][uTYP.L][0].frequency()

                    # Select carrier-phase frequency and iono frequency ratio
                    #
                    if f < nf:  # carrier
                        sig = obs.sig[sys][uTYP.L][f]
                        if sys == uGNSS.GLO:
                            freq = sig.frequency(self.nav.glo_ch[sat[j]])
                        else:
                            freq = sig.frequency()
                        mu = -(freq0/freq)**2
                    else:  # code
                        sig = obs.sig[sys][uTYP.C][f % nf]
                        if sys == uGNSS.GLO:
                            freq = sig.frequency(self.nav.glo_ch[sat[j]])
                        else:
                            freq = sig.frequency()
                        mu = +(freq0/freq)**2

                    # Skip edited observations
                    #
                    if np.any(self.nav.edt[sat[j]-1, :] > 0):
                        continue

                    # Skip invalid measurements
                    # NOTE: this additional test is included here,
                    #       since biases or antenna offsets may not be
                    #       available and this zdres()
                    #       returns zero observation residuals!
                    #
                    if y[i, f] == 0.0 or y[j, f] == 0.0:
                        continue

                    # Skip reference satellite i
                    #
                    if i == j:
                        continue

                    if mode == 0:  # DD
                        v[nv] = (y[i, f]-y[i+ns, f])-(y[j, f]-y[j+ns, f])
                    else:
                        #  Single-difference measurement
                        #
                        v[nv] = y[i, f] - y[j, f]

                    # SD line-of-sight vectors
                    #
                    H[nv, 0:3] = -e[i, :] + e[j, :]

                    if self.nav.ntrop > 0:  # tropo is estimated

                        # SD troposphere
                        #
                        _, mapfwi = tropmapf(
                            obs.t, pos, el[i], model=self.nav.trpModel)
                        _, mapfwj = tropmapf(
                            obs.t, pos, el[j], model=self.nav.trpModel)

                        idx_i = self.IT(self.nav.na)
                        H[nv, idx_i] = mapfwi - mapfwj
                        v[nv] -= (mapfwi - mapfwj)*x[idx_i]

                        if self.nav.monlevel > 2:
                            self.nav.fout.write(
                                fmt_ztd
                                .format(time2str(obs.t), idx_i, idx_i,
                                        (mapfwi - mapfwj),
                                        x[self.IT(self.nav.na)],
                                        np.sqrt(self.nav.P[
                                            self.IT(self.nav.na),
                                            self.IT(self.nav.na)])))

                    if self.nav.niono > 0:  # iono is estimated
                        # SD ionosphere
                        #
                        idx_i = self.II(sat[i], self.nav.na)
                        idx_j = self.II(sat[j], self.nav.na)
                        H[nv, idx_i] = +mu
                        H[nv, idx_j] = -mu
                        v[nv] -= mu*(x[idx_i] - x[idx_j])

                        if self.nav.monlevel > 2:
                            self.nav.fout.write(
                                fmt_ion
                                .format(time2str(obs.t),
                                        sat2id(sat[i]), sat2id(sat[j]),
                                        sig, idx_i, idx_j, mu,
                                        x[idx_i], x[idx_j],
                                        np.sqrt(self.nav.P[idx_i, idx_i]),
                                        np.sqrt(self.nav.P[idx_j, idx_j])))

                    # SD ambiguity
                    #
                    if f < nf:  # carrier-phase

                        idx_i = self.IB(sat[i], f, self.nav.na)
                        idx_j = self.IB(sat[j], f, self.nav.na)

                        if sys == uGNSS.GLO:
                            lami = sig.wavelength(self.nav.glo_ch[sat[i]])
                            lamj = sig.wavelength(self.nav.glo_ch[sat[j]])
                        else:
                            lami = sig.wavelength()
                            lamj = lami

                        H[nv, idx_i] = +lami
                        H[nv, idx_j] = -lamj
                        v[nv] -= lami*(x[idx_i] - x[idx_j])

                        # measurement variance
                        Ri[nv] = self.varerr(self.nav, el[i], f)
                        # measurement variance
                        Rj[nv] = self.varerr(self.nav, el[j], f)

                        self.nav.vsat[sat[i]-1, f] = 1
                        self.nav.vsat[sat[j]-1, f] = 1

                        if self.nav.monlevel > 2:
                            self.nav.fout.write(
                                fmt_amb
                                .format(time2str(obs.t),
                                        sat2id(sat[i]), sat2id(sat[j]),
                                        sig, idx_i, idx_j, lami, lamj,
                                        x[idx_i], x[idx_j],
                                        np.sqrt(self.nav.P[idx_i, idx_i]),
                                        np.sqrt(self.nav.P[idx_j, idx_j])))

                    else:  # pseudorange

                        # measurement variance
                        Ri[nv] = self.varerr(self.nav, el[i], f)
                        # measurement variance
                        Rj[nv] = self.varerr(self.nav, el[j], f)

                    if self.nav.monlevel > 1:
                        self.nav.fout.write(
                            fmt_res
                            .format(time2str(obs.t),
                                    sat2id(sat[i]), sat2id(sat[j]), sig,
                                    nv, v[nv],
                                    np.sqrt(Ri[nv]), np.sqrt(Rj[nv])))

                    nb[b] += 1  # counter for single-differences per signal
                    nv += 1  # counter for single-difference observations

                b += 1  # counter for signal (pseudrange+carrier-phase)

        v = np.resize(v, nv)
        H = np.resize(H, (nv, self.nav.nx))
        R = self.ddcov(nb, b, Ri, Rj, nv)

        return v, H, R

    def ddcov(self, nb, n, Ri, Rj, nv):
        """ DD measurement error covariance """
        R = np.zeros((nv, nv))
        k = 0
        for b in range(n):
            for i in range(nb[b]):
                for j in range(nb[b]):
                    R[k+i, k+j] = Ri[k+i]
                    if i == j:
                        R[k+i, k+j] += Rj[k+i]
            k += nb[b]
        return R

    def kfupdate(self, x, P, H, v, R):
        """
        Kalman filter measurement update.

        Parameters:
        x (ndarray): State estimate vector
        P (ndarray): State covariance matrix
        H (ndarray): Observation model matrix
        v (ndarray): Innovation vector
                     (residual between measurement and prediction)
        R (ndarray): Measurement noise covariance

        Returns:
        x (ndarray): Updated state estimate vector
        P (ndarray): Updated state covariance matrix
        S (ndarray): Innovation covariance matrix
        """

        PHt = P@H.T
        S = H@PHt+R
        K = PHt@np.linalg.inv(S)
        x += K@v
        # P = P - K@H@P
        IKH = np.eye(P.shape[0])-K@H
        P = IKH@P@IKH.T + K@R@K.T  # Joseph stabilized version

        return x, P, S

    def restamb(self, bias, nb):
        """ restore SD ambiguity """
        nv = 0
        xa = self.nav.x.copy()
        xa[0:self.nav.na] = self.nav.xa[0:self.nav.na]

        for m in range(uGNSS.GNSSMAX):
            for f in range(self.nav.nf):
                n = 0
                index = []
                for i in range(uGNSS.MAXSAT):
                    sys, _ = sat2prn(i+1)
                    if sys != m or self.nav.fix[i, f] != 2:
                        continue
                    index.append(self.IB(i+1, f, self.nav.na))
                    n += 1
                if n < 2:
                    continue
                xa[index[0]] = self.nav.x[index[0]]
                for i in range(1, n):
                    xa[index[i]] = xa[index[0]]-bias[nv]
                    nv += 1
        return xa

    def ddidx(self, nav, sat):
        """ index for SD to DD transformation matrix D """
        nb = 0
        n = uGNSS.MAXSAT
        na = nav.na
        ix = np.zeros((n, 2), dtype=int)
        nav.fix = np.zeros((n, nav.nf), dtype=int)
        for m in range(uGNSS.GNSSMAX):
            k = na
            for f in range(nav.nf):
                for i in range(k, k+n):
                    sat_i = i-k+1
                    sys, _ = sat2prn(sat_i)
                    if (sys != m):
                        continue
                    if sat_i not in sat or nav.x[i] == 0.0 \
                       or nav.vsat[sat_i-1, f] == 0:
                        continue
                    if nav.el[sat_i-1] >= nav.elmaskar:
                        nav.fix[sat_i-1, f] = 2
                        break
                    else:
                        nav.fix[sat_i-1, f] = 1
                for j in range(k, k+n):
                    sat_j = j-k+1
                    sys, _ = sat2prn(sat_j)
                    if (sys != m):
                        continue
                    if i == j or sat_j not in sat or nav.x[j] == 0.0 \
                       or nav.vsat[sat_j-1, f] == 0:
                        continue
                    if nav.el[sat_j-1] >= nav.elmaskar:
                        ix[nb, :] = [i, j]
                        nb += 1
                        nav.fix[sat_j-1, f] = 2
                k += n
        ix = np.resize(ix, (nb, 2))
        return ix

    def resamb_lambda(self, sat):
        """ resolve integer ambiguity using LAMBDA method """
        nx = self.nav.nx
        na = self.nav.na
        xa = np.zeros(na)
        ix = self.ddidx(self.nav, sat)
        nb = len(ix)
        if nb <= 0:
            print("no valid DD")
            return -1, -1

        # y=D*xc, Qb=D*Qc*D', Qab=Qac*D'
        y = self.nav.x[ix[:, 0]]-self.nav.x[ix[:, 1]]
        DP = self.nav.P[ix[:, 0], na:nx]-self.nav.P[ix[:, 1], na:nx]
        Qb = DP[:, ix[:, 0]-na]-DP[:, ix[:, 1]-na]
        Qab = self.nav.P[0:na, ix[:, 0]]-self.nav.P[0:na, ix[:, 1]]

        # MLAMBDA ILS
        b, s = mlambda(y, Qb)
        if s[0] <= 0.0 or s[1]/s[0] >= self.nav.thresar:
            self.nav.xa = self.nav.x[0:na].copy()
            self.nav.Pa = self.nav.P[0:na, 0:na].copy()
            bias = b[:, 0]
            y -= b[:, 0]
            K = Qab@np.linalg.inv(Qb)
            self.nav.xa -= K@y
            self.nav.Pa -= K@Qab.T

            # restore SD ambiguity
            xa = self.restamb(bias, nb)
        else:
            nb = 0

        return nb, xa

    def qcedit(self, obs, rs, dts, svh, rr=None):
        """ Coarse quality control and editing of observations """

        # Predicted position at next epoch
        #
        tt = timediff(obs.t, self.nav.t)
        if rr is None:
            rr_ = self.nav.x[0:3].copy()
            if self.nav.pmode > 0:
                rr_ += self.nav.x[3:6]*tt
        else:
            rr_ = rr

        # Geodetic position
        #
        pos = ecef2pos(rr_)

        # Total number of satellites
        #
        ns = uGNSS.MAXSAT

        # Reset previous editing results
        #
        self.nav.edt = np.zeros((ns, self.nav.nf), dtype=int)

        # Loop over all satellites
        #
        sat = []
        for i in range(ns):

            sat_i = i+1
            sys_i, _ = sat2prn(sat_i)

            if sat_i not in obs.sat:
                self.nav.edt[i, :] = 1
                continue

            # Check satellite exclusion
            #
            if sat_i in self.nav.excl_sat:
                self.nav.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - satellite excluded\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            j = np.where(obs.sat == sat_i)[0][0]

            # Check for valid orbit and clock offset
            #
            if np.isnan(rs[j, :]).any() or np.isnan(dts[j]):
                self.nav.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - invalid eph\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            # Check satellite health
            #
            if svh[j] > 0:
                self.nav.edt[i, :] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write("{}  {} - edit - satellite unhealthy\n"
                                        .format(time2str(obs.t),
                                                sat2id(sat_i)))
                continue

            # Check elevation angle
            #
            _, e = geodist(rs[j, :], rr_)
            _, el = satazel(pos, e)
            if el < self.nav.elmin:
                self.nav.edt[i][:] = 1
                if self.nav.monlevel > 0:
                    self.nav.fout.write(
                        "{}  {} - edit - low elevation {:5.1f} deg\n"
                        .format(time2str(obs.t), sat2id(sat_i),
                                np.rad2deg(el)))
                continue

            # Pseudorange, carrier-phase and C/N0 signals
            #
            sigsPR = obs.sig[sys_i][uTYP.C]
            sigsCP = obs.sig[sys_i][uTYP.L]
            sigsCN = obs.sig[sys_i][uTYP.S]

            # Loop over signals
            #
            for f in range(self.nav.nf):

                # Cycle  slip check by LLI
                #
                if obs.lli[j, f] == 1:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write("{}  {} - edit {:4s} - LLI\n"
                                            .format(time2str(obs.t),
                                                    sat2id(sat_i),
                                                    sigsCP[f].str()))
                    continue

                # Check for measurement consistency
                #
                if obs.P[j, f] == 0.0:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - invalid PR obs\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsPR[f].str()))
                    continue

                if obs.L[j, f] == 0.0:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - invalid CP obs\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsCP[f].str()))
                    continue

                # Check C/N0
                #
                cnr_min = self.nav.cnr_min_gpy \
                    if sigsCN[f].isGPS_PY() else self.nav.cnr_min
                if obs.S[j, f] < cnr_min:
                    self.nav.edt[i, f] = 1
                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - edit {:4s} - low C/N0 {:4.1f} dB-Hz\n"
                            .format(time2str(obs.t),
                                    sat2id(sat_i),
                                    sigsCN[f].str(),
                                    obs.S[j, f]))
                    continue

            # Store satellite which have passed all tests
            #
            if np.any(self.nav.edt[i, :] > 0):
                continue

            sat.append(sat_i)

        return np.array(sat, dtype=int)
        
    def gf_combination(self, obs, sat):
        # Loop over constellations
        #
        piono = np.zeros(uGNSS.MAXSAT)
        ciono = np.zeros(uGNSS.MAXSAT)
        
        for sys in obs.sig.keys():
            idx = self.sysidx(sat, sys)
            if len(idx) == 0:
                continue
            
            freq0 = obs.sig[sys][uTYP.L][0].frequency()
            freq1 = obs.sig[sys][uTYP.L][1].frequency()
            
            lam0 = rCST.CLIGHT/freq0
            lam1 = rCST.CLIGHT/freq1
            
            piono[idx] = lam0 * obs.L[idx, 0] - lam1 * obs.L[idx, 1]
            ciono[idx] = obs.P[idx, 0] - obs.P[idx, 1]
            
            piono = -piono / (1 - freq0*freq0/freq1/freq1)
            ciono = ciono / (1 - freq0*freq0/freq1/freq1)
        
        return piono, ciono

    def base_process(self, obs, obsb, rs, dts, svh):
        """ processing for base station in RTK
            (implemented in rtkpos) """
        return None, None, None, None

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        """
        PPP/PPP-RTK/RTK positioning
        """

        # Skip empty epochs
        #
        if len(obs.sat) == 0:
            return

        # GNSS satellite positions, velocities and clock offsets
        # for all satellite in RINEX observations
        #
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)

        if nsat < 6:
            print(" too few satellites < 6: nsat={:d}".format(nsat))
            return

        # Editing of observations
        #
        sat_ed = self.qcedit(obs, rs, dts, svh)

        if obsb is None:  # PPP/PPP-RTK
            # Select satellites having passed quality control
            #
            # index of valid sats in obs.sat
            iu = np.where(np.isin(obs.sat, sat_ed))[0]
            ns = len(iu)
            y = np.zeros((ns, self.nav.nf*2))
            e = np.zeros((ns, 3))

            obs_ = obs
        else:  # RTK
            y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
            ns = len(iu)

        if ns < 6:
            print(" too few satellites < 6: ns={:d}".format(ns))
            return

        # Kalman filter time propagation, initialization of ambiguities
        # and iono
        #
        self.udstate(obs_)

        xa = np.zeros(self.nav.nx)
        xp = self.nav.x.copy()

        # Non-differential residuals
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        sat = obs.sat[iu]
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        el = elu[iu]

        # Store reduced satellite list
        # NOTE: where are working on a reduced list of observations
        # from here on
        #
        self.nav.sat = sat
        self.nav.el[sat-1] = el  # needed in rtk.ddidx()
        self.nav.y = y
        ns = len(sat)

        # Check if observations of at least 6 satellites are left over
        # after editing
        #
        ny = y.shape[0]
        if ny < 6:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 5
            return -1

        # SD residuals
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        Pp = self.nav.P.copy()

        # Kalman filter measurement update
        #
        xp, Pp, _ = self.kfupdate(xp, Pp, H, v, R)

        # Non-differential residuals after measurement update
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])
        y = yu[iu, :]
        e = eu[iu, :]
        ny = y.shape[0]
        if ny < 6:
            return -1

        # Residuals for float solution
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        if self.valpos(v, R):
            self.nav.x = xp
            self.nav.P = Pp
            self.nav.ns = 0
            
            self.nav.piano, self.nav.ciano = self.gf_combination(obs, sat)
            
            for i in range(ns):
                j = sat[i]-1
                for f in range(self.nav.nf):
                    if self.nav.vsat[j, f] == 0:
                        continue
                    self.nav.outc[j, f] = 0
                    if f == 0:
                        self.nav.ns += 1
        else:
            self.nav.smode = 0

        self.nav.smode = 5  # 4: fixed ambiguities, 5: float ambiguities

        if self.nav.armode > 0:
            nb, xa = self.resamb_lambda(sat)
            if nb > 0:
                # Use position with fixed ambiguities xa
                yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xa[0:3])
                y = yu[iu, :]
                e = eu[iu, :]
                v, H, R = self.sdres(obs, xa, y, e, sat, el)
                # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
                if self.valpos(v, R):
                    self.nav.smode = 4           # fix

        # Store epoch for solution
        #
        self.nav.t = obs.t

        return 0
