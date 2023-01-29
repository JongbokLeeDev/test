# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

from cssrlib.gnss import Nav, epoch2time, time2epoch, timeadd, \
    id2sat, timediff, gtime_t, uGNSS, str2time, sat2prn, rCST, rSIG
from cssrlib.rinex import rnxdec
import numpy as np
from math import pow, sin, cos

NMAX = 10
MAXDTE = 900.0
EXTERR_CLK = 1e-3
EXTERR_EPH = 5e-7

class peph_t:    
    def __init__(self, time = None):
        if time is not None:
            self.time = time
        else:
            self.time = gtime_t()
        self.pos = np.zeros((uGNSS.MAXSAT,4))
        self.vel = np.zeros((uGNSS.MAXSAT,4))
        self.std = np.zeros((uGNSS.MAXSAT,4))
        self.vst = np.zeros((uGNSS.MAXSAT,4))
    
class peph:
    nsat = 0
    nep  = 0
    t0 = None
    week0 = -1
    svid = None
    acc  = None
    svpos = None
    svclk = None
    svvel = None
    status = 0
    scl = [0.0,0.0]
    
    def __init__(self):
        self.t = None
        self.nmax = 24*12
        
    def parse_satlist(self,line):
        n = len(line[9:])//3
        for k in range(n):
            svid = line[9+3*k:12+3*k]
            if int(svid[1:])>0:
                self.sat[self.cnt]=id2sat(svid)
                self.cnt+=1

    def parse_acclist(self,line):
        n = len(line[9:])//3
        for k in range(n):
            acc = int(line[9+3*k:12+3*k])
            if self.cnt<self.nsat:
                self.acc[self.cnt] = acc
            self.cnt+=1
                                    
    def parse_sp3(self,fname, nav, opt=0):
        ver_t = ['c','d']
        self.status = 0
        v = False
        with open(fname,"r") as fh:
            for line in fh:
                if line[0:3]=='EOF': # end of record
                    break
                if line[0:2]=='/*': # skip comment
                    continue
                if line[0:2]=='* ': # header of body part
                    self.status = 10

                if self.status== 0:
                    if line[0]!='#':
                        break
                    self.ver=line[1]
                    if self.ver not in ver_t:
                        print("invalid version: {:s}".format(self.ver))
                        break
                    self.flag=line[2]
                    self.t0 = str2time(line,3,27)                    
                    self.status = 1
                elif self.status == 1:
                    if line[0:2]!='##':
                        break
                    self.week0 = int(line[3:7])
                    print("week={:4d}".format(self.week0))                    
                    self.status = 2
                elif self.status == 2:
                    if line[0:2]=='+ ':
                        self.cnt = 0
                        self.nsat = int(line[3:6])
                        self.sat = np.zeros(self.nsat,dtype=int)
                        self.acc = np.zeros(self.nsat,dtype=int)
                        self.parse_satlist(line)
                        self.status = 3
                        continue
                elif self.status == 3:
                    if line[0:2]=='++':
                        self.cnt = 0
                        self.status = 4
                        self.parse_acclist(line)
                        continue
                    self.parse_satlist(line)                                            
                elif self.status == 4:
                    if line[0:2]=='%c':
                        # two %c, two %f, two %i records
                        # Columns 10-12 in the first %c record define
                        # what time system is used for the date/times 
                        # in the ephemeris.
                        # c4-5 File type
                        # c10-12 Time System
                        line = fh.readline() # %c
                        line = fh.readline() # %f
                        if line[0:2]!='%f':
                            break
                        # Base for Pos/Vel  (mm or 10**-4 mm/sec)
                        # Base for Clk/Rate (psec or 10**-4 psec/sec)
                        self.scl[0] = float(line[3:13])
                        self.scl[1] = float(line[14:26])                        
                        self.status = 10
                        for k in range(3):
                            line = fh.readline()
                        continue
                    self.parse_acclist(line)                   
                    
                if self.status==10: # body
                    if line[0:2]=='* ': # epoch header
                        v = False
                        nav.ne += 1
                        self.cnt = 0
                        peph = peph_t(str2time(line,3,27))                        
                        #print("{:4.0f}/{:02.0f}/{:02.0f} {:2.0f}:{:2.0f}:{:5.2f}"\
                        #      .format(ep[0],ep[1],ep[2],ep[3],ep[4],ep[5]))
 
                        for k in range(self.nsat):
                            line = fh.readline()
                            if line[0]!='P' and line[0]!='V':
                                continue
                                                        
                            svid = line[1:4]
                            sat_=id2sat(svid)
                        
                            #clk_ev   = line[74] # clock event flag
                            #clk_pred = line[75] # clock pred. flag
                            #mnv_flag = line[78] # maneuver flag
                            #orb_pred = line[79] # orbit pred. flag
                            pred_c = len(line)>=76 and line[75]=='P'
                            pred_o = len(line)>=80 and line[79]=='P'
                        
                            # x,y,z[km],clock[usec]
                            for j in range(4):
                                if j<3 and (opt&1) and pred_o:
                                    continue
                                if j<3 and (opt&2) and not pred_o:
                                    continue
                                if j==3 and (opt&1) and pred_c:
                                    continue
                                if j==3 and (opt&2) and not pred_c:
                                    continue                            
                            
                                val = float(line[4+j*14:18+j*14])
                                if val!=0.0 and abs(val-999999.999999)>=1e-6:
                                    scl = 1e3 if j<3 else 1e-6
                                    if line[0] == 'P':
                                        v = True
                                        peph.pos[sat_-1,j] = val*scl
                                    elif v:
                                        peph.vel[sat_-1,j] = val*scl*1e-4
                                                 
                            if len(line)>=74:
                                for j in range(4):
                                    if j<3:
                                        slen, scl, ofst = 2, 1e-3, 0
                                    else:
                                        slen, scl, ofst = 3, 1e-12, 1
                                    s = line[61+j*3:61+j*3+slen]
                                    std = int(s) if s[-1] != ' ' else 0
                                    if self.scl[ofst]>0.0 and std>0.0:
                                        v = pow(self.scl[ofst],std)*scl
                                        if line[0] == 'P':
                                            peph.std[sat_-1,j] = v
                                        else:
                                            peph.vst[sat_-1,j] = v*1e-4
                    if v:
                        nav.peph.append(peph)

        return nav

    def interppol(self, x, y, n):
        for j in range(1,n):
            for i in range(n-j):
                y[i] = (x[i+j]*y[i]-x[i]*y[i+1])/(x[i+j]-x[i])
        return y[0]
          
    def pephpos(self, time, sat, nav, vare=False, varc=False):
        rs = np.zeros(3)
        dts = np.zeros(2)
        t = np.zeros(NMAX+1)
        p = np.zeros((3,NMAX+1))
        
        if nav.ne<NMAX+1 or \
            timediff(time, nav.peph[0].time)<-MAXDTE or \
            timediff(time, nav.peph[nav.ne-1].time)>MAXDTE:
            return False
        i, j = 0, nav.ne-1
        while i<j:
            k = (i+j)//2
            if timediff(nav.peph[k].time,time)<0.0:
                i=k+1
            else:
                j=k
        index = 0 if i<=0 else i-1
        
        i=index-(NMAX+1)//2
        if i<0:
            i=0
        elif i+NMAX>=nav.ne:
            i=nav.ne-NMAX-1
                
        for j in range(NMAX+1):
            t[j] = timediff(nav.peph[i+j].time,time)
            if np.linalg.norm(nav.peph[i+j].pos[sat-1,:])<=0.0:
                return False
        
        for j in range(NMAX+1):
            pos = nav.peph[i+j].pos[sat-1,:]
            sinl = sin(rCST.OMGE*t[j])
            cosl = cos(rCST.OMGE*t[j])            
            p[0,j] = cosl*pos[0]-sinl*pos[1]
            p[1,j] = sinl*pos[0]+cosl*pos[1]
            p[2,j] = pos[2]
            
        for i in range(3):
            rs[i] = self.interppol(t,p[i,:],NMAX+1)
        
        p_ = nav.peph[index:index+2]
        
        if vare:
            s = np.zeros(3)
            for i in range(3):
                s[i] = p_[0].std[sat-1,i]
            std = np.linalg.norm(s)
            if t[0]>0.0:
                std += EXTERR_EPH*(t[0]**2)/2.0
            elif t[NMAX]<0.0:
                std += EXTERR_EPH*(t[NMAX]**2)/2.0
            vare = std**2
        
        t[0] = timediff(time,p_[0].time)
        t[1] = timediff(time,p_[1].time)
          
        c = [p_[0].pos[sat-1,3],p_[1].pos[sat-1,3]]
                      
        if t[0]<=0.0:
            dts[0]=c[0]
            if dts[0]!=0.0:
                std = p_[0].std[sat-1,3]*rCST.CLIGHT-EXTERR_CLK*t[0]
        elif t[1]>=0.0:
            dts[0]=c[1]
            if dts[0]!=0.0:
                std = p_[1].std[sat-1,3]*rCST.CLIGHT-EXTERR_CLK*t[1]
        elif c[0]!=0.0 and c[1]!=0.0:
            dts[0]=(c[1]*t[0]-c[0]*t[1])/(t[0]-t[1])
            i= 0 if t[0]<-t[1] else 1
            std= p_[i].std[sat-1,3]+EXTERR_CLK*abs(t[i])
        else:
            dts[0] = 0.0

        if varc:
            varc = std**2
        
        return rs, dts, vare, varc

    def pephclk(self, time, sat, nav, varc=False):
        dts = np.zeros(2)
        t = np.zeros(NMAX+1)
        
        if nav.nc<2 or \
            timediff(time,nav.pclk[0].time)<-MAXDTE or \
            timediff(time,nav.pclk[nav.nc-1].time)>MAXDTE:
            return False
        i, j = 0, nav.nc-1
        while i<j:
            k = (i+j)//2
            if timediff(nav.pclk[k].time,time)<0.0:
                i=k+1
            else:
                j=k
        index = 0 if i<=0 else i-1
        p_ = nav.pclk[index:index+2]
        
        t[0] = timediff(time,p_[0].time)
        t[1] = timediff(time,p_[1].time)
        
        c = [p_[0].clk[sat-1],p_[1].clk[sat-1]]
        
        if t[0]<=0.0:
            dts[0]=c[0]
            if dts[0]==0.0:
                return False
            std = p_[0].std[sat-1]*rCST.CLIGHT-EXTERR_CLK*t[0]
        elif t[1]>=0.0:
            dts[0]=c[1]
            if dts[0]==0.0:
                return False
            std = p_[1].std[sat-1]*rCST.CLIGHT-EXTERR_CLK*t[1]
        elif c[0]!=0.0 and c[1]!=0.0:
            dts[0]=(c[1]*t[0]-c[0]*t[1])/(t[0]-t[1])
            i= 0 if t[0]<-t[1] else 1
            std= p_[i].std[sat-1]+EXTERR_CLK*abs(t[i])
        else:
            return False

        if varc:
            varc = std**2
        
        return dts,varc

    def peph2pos(self, time, sat, nav, var=False):
        tt=1e-3
        rss, dtss, vare, varc = self.pephpos(time,sat,nav)
        if nav.nc>=2:
            dtss, varc = self.pephclk(time, sat, nav)
        time_tt = timeadd(time,tt)
        rst, dtst, _, _ = self.pephpos(time_tt,sat,nav)
        if nav.nc>=2:
            dtst, _ = self.pephclk(time_tt, sat, nav)            
        
        dant = satantoff(time, rss, sat, nav)
        rs = np.zeros(6)
        dts = np.zeros(2)
        
        rs[0:3] = rss + dant
        rs[3:6] = (rst-rss)/tt
        
        if dtss[0]!=0.0:
            dts[0]=dtss[0]-2.0*(rs[0:3]@rs[3:6])/(rCST.CLIGHT**2)
            dts[1]=(dtst[0]-dtss[0])/tt
        else:
            dts = dtss

        if var:
            var = vare+varc
            
        return rs, dts, var
   
NFREQ = 3

class pcv_t():
    def __init__(self):
        self.sat = 0
        self.code = ''
        self.type = ''
        self.ts = gtime_t()
        self.te = gtime_t()
        self.off = np.zeros((NFREQ,3))
        self.var = np.zeros((NFREQ,19))
        self.zen = [0,0,0]
        self.nv = 0
        self.dazi = 0.0
        


def readpcv(fname):
    pcvs = []
    state = False
    freq = 0
    freqs = [1,2,5,6,7,8,0]
    
    with open(fname,"r") as fh:
        for line in fh:
            if len(line)<60 or "COMMENT" in line[60:]:
                continue
            if "START OF ANTENNA" in line[60:]:
                pcv = pcv_t()
                state = True
            elif "END OF ANTENNA" in line[60:]:
                pcvs.append(pcv)
                state = False
            if not state:
                continue
            if "TYPE / SERIAL NO" in line[60:]:
                pcv.type = line[0:20]
                pcv.code = line[20:40]
                if pcv.code[3:11] == "        ":
                    pcv.sat = id2sat(pcv.code)
            elif "VALID FROM" in line[60:]:
                pcv.ts = str2time(line,2,40)
            elif "VALID UNTIL" in line[60:]:
                pcv.te = str2time(line,2,40)
            elif "START OF FREQUENCY" in line[60:]:
                f = int(line[4:6])
                for i in range(NFREQ):
                    if freqs[i]==f:
                        break
                if i<NFREQ:
                    freq=i+1        
            elif "END OF FREQUENCY" in line[60:]:
                freq = 0
            elif "NORTH / EAST / UP" in line[60:]:
                if freq<1 or NFREQ<freq:
                    continue
                neu = [float(x)*1e-3 for x in line[3:30].split()]
                pcv.off[freq-1,0] = neu[0] if pcv.sat>0 else neu[1]
                pcv.off[freq-1,1] = neu[1] if pcv.sat>0 else neu[0]
                pcv.off[freq-1,2] = neu[2]
            elif "ZEN1 / ZEN2 / DZEN" in line[60:]:
                pcv.zen = [float(x) for x in line[3:20].split()]
                pcv.nv = int((pcv.zen[1]-pcv.zen[0]+1)/pcv.zen[2])
            elif "DAZI" in line[60:]:
                pcv.dazi = float(line[3:8])
            elif "NOAZI" in line[3:8]:
                if freq<1 or NFREQ<freq:
                    continue
                var = [float(x) for x in line[8:].split()]
                n = min(len(var),19)
                if n<19:
                    pcv.var[freq-1,n:] = var[n-1]                    
                pcv.var[freq-1,:n] = var[0:n]

    return pcvs

def searchpcv(sat,time,pcvs):
    n = len(pcvs)
    for i in range(n):
        pcv = pcvs[i]
        if pcv.sat != sat:
            continue
        if pcv.ts.time!=0 and timediff(pcv.ts,time)>0.0:
            continue
        if pcv.te.time!=0 and timediff(pcv.te,time)<0.0:
            continue
        return pcv
    return False


leaps_ = [[2017,1,1,0,0,0,-18],
          [2015,7,1,0,0,0,-17],
          [2012,7,1,0,0,0,-16],
          [2009,1,1,0,0,0,-15],
          [2006,1,1,0,0,0,-14],
          [1999,1,1,0,0,0,-13],
          [1997,7,1,0,0,0,-12],
          [1996,1,1,0,0,0,-11],
          [1994,7,1,0,0,0,-10],
          [1993,7,1,0,0,0, -9],
          [1992,7,1,0,0,0, -8],
          [1991,1,1,0,0,0, -7],
          [1990,1,1,0,0,0, -6],
          [1988,1,1,0,0,0, -5],
          [1985,7,1,0,0,0, -4],
          [1983,7,1,0,0,0, -3],
          [1982,7,1,0,0,0, -2],
          [1981,7,1,0,0,0, -1]]

def gpst2utc(t:gtime_t):
    for i in range(len(leaps_)):
        tu = timeadd(t,leaps_[i][6])
        if timediff(tu,epoch2time(leaps_[i]))>=0.0:
            return tu
    return t
                
def utc2gpst(t:gtime_t):
    for i in range(len(leaps_)):
        if timediff(t,epoch2time(leaps_[i]))>=0.0:
            return timeadd(t,-leaps_[i][6])
    return t

def Rx(t):
    ct, st = cos(t), sin(t)
    return np.array([[1.0,0.0,0.0],[0.0,ct,st],[0.0,-st,ct]])

def Ry(t):
    ct, st = cos(t), sin(t)
    return np.array([[ct,0.0,-st],[0.0,1.0,0.0],[st,0.0,ct]])

def Rz(t):
    ct, st = cos(t), sin(t)
    return np.array([[ct,st,0.0],[-st,ct,0.0],[0.0,0.0,1.0]])

def nut_iau1980(t,f):
    nut = np.array([
        [   0,   0,   0,   0,   1, -6798.4, -171996, -174.2, 92025,   8.9],
        [   0,   0,   2,  -2,   2,   182.6,  -13187,   -1.6,  5736,  -3.1],
        [   0,   0,   2,   0,   2,    13.7,   -2274,   -0.2,   977,  -0.5],
        [   0,   0,   0,   0,   2, -3399.2,    2062,    0.2,  -895,   0.5],
        [   0,  -1,   0,   0,   0,  -365.3,   -1426,    3.4,    54,  -0.1],
        [   1,   0,   0,   0,   0,    27.6,     712,    0.1,    -7,   0.0],
        [   0,   1,   2,  -2,   2,   121.7,    -517,    1.2,   224,  -0.6],
        [   0,   0,   2,   0,   1,    13.6,    -386,   -0.4,   200,   0.0],
        [   1,   0,   2,   0,   2,     9.1,    -301,    0.0,   129,  -0.1],
        [   0,  -1,   2,  -2,   2,   365.2,     217,   -0.5,   -95,   0.3],
        [  -1,   0,   0,   2,   0,    31.8,     158,    0.0,    -1,   0.0],
        [   0,   0,   2,  -2,   1,   177.8,     129,    0.1,   -70,   0.0],
        [  -1,   0,   2,   0,   2,    27.1,     123,    0.0,   -53,   0.0],
        [   1,   0,   0,   0,   1,    27.7,      63,    0.1,   -33,   0.0],
        [   0,   0,   0,   2,   0,    14.8,      63,    0.0,    -2,   0.0],
        [  -1,   0,   2,   2,   2,     9.6,     -59,    0.0,    26,   0.0],
        [  -1,   0,   0,   0,   1,   -27.4,     -58,   -0.1,    32,   0.0],
        [   1,   0,   2,   0,   1,     9.1,     -51,    0.0,    27,   0.0],
        [  -2,   0,   0,   2,   0,  -205.9,     -48,    0.0,     1,   0.0],
        [  -2,   0,   2,   0,   1,  1305.5,      46,    0.0,   -24,   0.0],
        [   0,   0,   2,   2,   2,     7.1,     -38,    0.0,    16,   0.0],
        [   2,   0,   2,   0,   2,     6.9,     -31,    0.0,    13,   0.0],
        [   2,   0,   0,   0,   0,    13.8,      29,    0.0,    -1,   0.0],
        [   1,   0,   2,  -2,   2,    23.9,      29,    0.0,   -12,   0.0],
        [   0,   0,   2,   0,   0,    13.6,      26,    0.0,    -1,   0.0],
        [   0,   0,   2,  -2,   0,   173.3,     -22,    0.0,     0,   0.0],
        [  -1,   0,   2,   0,   1,    27.0,      21,    0.0,   -10,   0.0],
        [   0,   2,   0,   0,   0,   182.6,      17,   -0.1,     0,   0.0],
        [   0,   2,   2,  -2,   2,    91.3,     -16,    0.1,     7,   0.0],
        [  -1,   0,   0,   2,   1,    32.0,      16,    0.0,    -8,   0.0],
        [   0,   1,   0,   0,   1,   386.0,     -15,    0.0,     9,   0.0],
        [   1,   0,   0,  -2,   1,   -31.7,     -13,    0.0,     7,   0.0],
        [   0,  -1,   0,   0,   1,  -346.6,     -12,    0.0,     6,   0.0],
        [   2,   0,  -2,   0,   0, -1095.2,      11,    0.0,     0,   0.0],
        [  -1,   0,   2,   2,   1,     9.5,     -10,    0.0,     5,   0.0],
        [   1,   0,   2,   2,   2,     5.6,      -8,    0.0,     3,   0.0],
        [   0,  -1,   2,   0,   2,    14.2,      -7,    0.0,     3,   0.0],
        [   0,   0,   2,   2,   1,     7.1,      -7,    0.0,     3,   0.0],
        [   1,   1,   0,  -2,   0,   -34.8,      -7,    0.0,     0,   0.0],
        [   0,   1,   2,   0,   2,    13.2,       7,    0.0,    -3,   0.0],
        [  -2,   0,   0,   2,   1,  -199.8,      -6,    0.0,     3,   0.0],
        [   0,   0,   0,   2,   1,    14.8,      -6,    0.0,     3,   0.0],
        [   2,   0,   2,  -2,   2,    12.8,       6,    0.0,    -3,   0.0],
        [   1,   0,   0,   2,   0,     9.6,       6,    0.0,     0,   0.0],
        [   1,   0,   2,  -2,   1,    23.9,       6,    0.0,    -3,   0.0],
        [   0,   0,   0,  -2,   1,   -14.7,      -5,    0.0,     3,   0.0],
        [   0,  -1,   2,  -2,   1,   346.6,      -5,    0.0,     3,   0.0],
        [   2,   0,   2,   0,   1,     6.9,      -5,    0.0,     3,   0.0],
        [   1,  -1,   0,   0,   0,    29.8,       5,    0.0,     0,   0.0],
        [   1,   0,   0,  -1,   0,   411.8,      -4,    0.0,     0,   0.0],
        [   0,   0,   0,   1,   0,    29.5,      -4,    0.0,     0,   0.0],
        [   0,   1,   0,  -2,   0,   -15.4,      -4,    0.0,     0,   0.0],
        [   1,   0,  -2,   0,   0,   -26.9,       4,    0.0,     0,   0.0],
        [   2,   0,   0,  -2,   1,   212.3,       4,    0.0,    -2,   0.0],
        [   0,   1,   2,  -2,   1,   119.6,       4,    0.0,    -2,   0.0],
        [   1,   1,   0,   0,   0,    25.6,      -3,    0.0,     0,   0.0],
        [   1,  -1,   0,  -1,   0, -3232.9,      -3,    0.0,     0,   0.0],
        [  -1,  -1,   2,   2,   2,     9.8,      -3,    0.0,     1,   0.0],
        [   0,  -1,   2,   2,   2,     7.2,      -3,    0.0,     1,   0.0],
        [   1,  -1,   2,   0,   2,     9.4,      -3,    0.0,     1,   0.0],
        [   3,   0,   2,   0,   2,     5.5,      -3,    0.0,     1,   0.0],
        [  -2,   0,   2,   0,   2,  1615.7,      -3,    0.0,     1,   0.0],
        [   1,   0,   2,   0,   0,     9.1,       3,    0.0,     0,   0.0],
        [  -1,   0,   2,   4,   2,     5.8,      -2,    0.0,     1,   0.0],
        [   1,   0,   0,   0,   2,    27.8,      -2,    0.0,     1,   0.0],
        [  -1,   0,   2,  -2,   1,   -32.6,      -2,    0.0,     1,   0.0],
        [   0,  -2,   2,  -2,   1,  6786.3,      -2,    0.0,     1,   0.0],
        [  -2,   0,   0,   0,   1,   -13.7,      -2,    0.0,     1,   0.0],
        [   2,   0,   0,   0,   1,    13.8,       2,    0.0,    -1,   0.0],
        [   3,   0,   0,   0,   0,     9.2,       2,    0.0,     0,   0.0],
        [   1,   1,   2,   0,   2,     8.9,       2,    0.0,    -1,   0.0],
        [   0,   0,   2,   1,   2,     9.3,       2,    0.0,    -1,   0.0],
        [   1,   0,   0,   2,   1,     9.6,      -1,    0.0,     0,   0.0],
        [   1,   0,   2,   2,   1,     5.6,      -1,    0.0,     1,   0.0],
        [   1,   1,   0,  -2,   1,   -34.7,      -1,    0.0,     0,   0.0],
        [   0,   1,   0,   2,   0,    14.2,      -1,    0.0,     0,   0.0],
        [   0,   1,   2,  -2,   0,   117.5,      -1,    0.0,     0,   0.0],
        [   0,   1,  -2,   2,   0,  -329.8,      -1,    0.0,     0,   0.0],
        [   1,   0,  -2,   2,   0,    23.8,      -1,    0.0,     0,   0.0],
        [   1,   0,  -2,  -2,   0,    -9.5,      -1,    0.0,     0,   0.0],
        [   1,   0,   2,  -2,   0,    32.8,      -1,    0.0,     0,   0.0],
        [   1,   0,   0,  -4,   0,   -10.1,      -1,    0.0,     0,   0.0],
        [   2,   0,   0,  -4,   0,   -15.9,      -1,    0.0,     0,   0.0],
        [   0,   0,   2,   4,   2,     4.8,      -1,    0.0,     0,   0.0],
        [   0,   0,   2,  -1,   2,    25.4,      -1,    0.0,     0,   0.0],
        [  -2,   0,   2,   4,   2,     7.3,      -1,    0.0,     1,   0.0],
        [   2,   0,   2,   2,   2,     4.7,      -1,    0.0,     0,   0.0],
        [   0,  -1,   2,   0,   1,    14.2,      -1,    0.0,     0,   0.0],
        [   0,   0,  -2,   0,   1,   -13.6,      -1,    0.0,     0,   0.0],
        [   0,   0,   4,  -2,   2,    12.7,       1,    0.0,     0,   0.0],
        [   0,   1,   0,   0,   2,   409.2,       1,    0.0,     0,   0.0],
        [   1,   1,   2,  -2,   2,    22.5,       1,    0.0,    -1,   0.0],
        [   3,   0,   2,  -2,   2,     8.7,       1,    0.0,     0,   0.0],
        [  -2,   0,   2,   2,   2,    14.6,       1,    0.0,    -1,   0.0],
        [  -1,   0,   0,   0,   2,   -27.3,       1,    0.0,    -1,   0.0],
        [   0,   0,  -2,   2,   1,  -169.0,       1,    0.0,     0,   0.0],
        [   0,   1,   2,   0,   1,    13.1,       1,    0.0,     0,   0.0],
        [  -1,   0,   4,   0,   2,     9.1,       1,    0.0,     0,   0.0],
        [   2,   1,   0,  -2,   0,   131.7,       1,    0.0,     0,   0.0],
        [   2,   0,   0,   2,   0,     7.1,       1,    0.0,     0,   0.0],
        [   2,   0,   2,  -2,   1,    12.8,       1,    0.0,    -1,   0.0],
        [   2,   0,  -2,   0,   1,  -943.2,       1,    0.0,     0,   0.0],
        [   1,  -1,   0,  -2,   0,   -29.3,       1,    0.0,     0,   0.0],
        [  -1,   0,   0,   1,   1,  -388.3,       1,    0.0,     0,   0.0],
        [  -1,  -1,   0,   2,   1,    35.0,       1,    0.0,     0,   0.0],
        [   0,   1,   0,   1,   0,    27.3,       1,    0.0,     0,   0.0]        
        ])

    dpsi=deps=0.0
    for i in range(106):
        ang=0.0
        for j in range(5):
            ang+=nut[i][j]*f[j]
        dpsi+=(nut[i][6]+nut[i][7]*t)*sin(ang)
        deps+=(nut[i][8]+nut[i][9]*t)*cos(ang)
    dpsi *= 1E-4*rCST.AS2R # 0.1 mas -> rad
    deps *= 1E-4*rCST.AS2R
    return dpsi, deps

def time2sec(time):
    ep = time2epoch(time)
    sec=ep[3]*3600.0+ep[4]*60.0+ep[5]
    ep[3]=ep[4]=ep[5]=0.0
    day=epoch2time(ep)
    return sec, day
    
def utc2gmst(t:gtime_t,ut1_utc):
    ep2000=[2000,1,1,12,0,0]
    
    tut=timeadd(t,ut1_utc)
    ut, tut0 =time2sec(tut)
    t1=timediff(tut0,epoch2time(ep2000))/86400.0/36525.0;
    t2=t1*t1
    t3=t2*t1
    gmst0=24110.54841+8640184.812866*t1+0.093104*t2-6.2E-6*t3
    gmst=gmst0+1.002737909350795*ut
    
    return np.fmod(gmst,86400.0)*np.pi/43200.0 # 0 <= gmst <= 2*PI

def eci2ecef(tutc,erpv):
    ep2000 = [2000,1,1,12,0,0]
    
    tgps = utc2gpst(tutc)
    t = (timediff(tgps,epoch2time(ep2000))+19.0+32.184)/86400.0/36525.0
    t2 = t**2
    t3 = t2*t
    f = ast_args(t)
    
    ze =(2306.2181*t+0.20188*t2+0.017998*t3)*rCST.AS2R
    th=(2004.3109*t-0.42665*t2-0.041833*t3)*rCST.AS2R
    z =(2306.2181*t+1.09468*t2+0.018203*t3)*rCST.AS2R
    eps=(84381.448-46.8150*t-0.00059*t2+0.001813*t3)*rCST.AS2R
    P = Rz(-z)@Ry(th)@Rz(-ze)
    
    dpsi, deps = nut_iau1980(t,f)
    N = Rx(-eps-deps)@Rz(-dpsi)@Rx(eps)
    
    gmst_ = utc2gmst(tutc,erpv[2])
    gast = gmst_+dpsi*np.cos(eps)
    gast += (0.00264*np.sin(f[4])+0.000063*np.sin(2.0*f[4]))*rCST.AS2R
    
    W = Ry(-erpv[0])@Rx(-erpv[1])
    U = W@Rz(gast)@N@P

    return U, gmst_

def ast_args(t):
    ''' astronomical arguments: f={l,l',F,D,OMG} (rad) '''
    # coefficients for iau 1980 nutation
    fc = np.array([[134.96340251, 1717915923.2178,  31.8792,  0.051635, -0.00024470],
                   [357.52910918,  129596581.0481,  -0.5532,  0.000136, -0.00001149],
                   [ 93.27209062, 1739527262.8478, -12.7512, -0.001037,  0.00000417],
                   [297.85019547, 1602961601.2090,  -6.3706,  0.006593, -0.00003169],
                   [125.04455501,   -6962890.2665,   7.4722,  0.007702, -0.00005939]])
    f = np.zeros(5)
    tt = np.zeros(4)
    tt[0] = t
    for i in range(1,4):
        tt[i] = tt[i-1]*t
    for i in range(5):
        f[i] = fc[i][0]*3600.0
        for j in range(4):
            f[i]+=fc[i][j+1]*tt[j]
        f[i] = np.fmod(f[i]*rCST.AS2R,2.0*np.pi)
    return f
    
def sunmoonpos_eci(tut,rsun=False,rmoon=False):
    ep2000 = [2000,1,1,12,0,0]
    t = timediff(tut,epoch2time(ep2000))/86400.0/36525.0
    f = ast_args(t)
    eps=23.439291-0.0130042*t
    sine, cose = sin(eps*rCST.D2R), cos(eps*rCST.D2R)
    if rsun:
        Ms=357.5277233+35999.05034*t
        ls=280.460+36000.770*t+1.914666471*sin(Ms*rCST.D2R)+0.019994643*sin(2.0*Ms*rCST.D2R)
        rs=rCST.AU*(1.000140612-0.016708617*cos(Ms*rCST.D2R)-0.000139589*cos(2.0*Ms*rCST.D2R))
        sinl,cosl = sin(ls*rCST.D2R),cos(ls*rCST.D2R)
        rsun = rs*np.array([cosl,cose*sinl,sine*sinl])
    if rmoon:
        lm=218.32+481267.883*t+6.29*sin(f[0])-1.27*sin(f[0]-2.0*f[3]) \
            +0.66*sin(2.0*f[3])+0.21*sin(2.0*f[0])-0.19*sin(f[1])-0.11*sin(2.0*f[2])        
        pm=5.13*sin(f[2])+0.28*sin(f[0]+f[2])-0.28*sin(f[2]-f[0])-0.17*sin(f[2]-2.0*f[3])
        rm=rCST.RE_WGS84/sin((0.9508+0.0518*cos(f[0])+0.0095*cos(f[0]-2.0*f[3]) \
                         +0.0078*cos(2.0*f[3])+0.0028*cos(2.0*f[0]))*rCST.D2R)
        sinl, cosl = sin(lm*rCST.D2R), cos(lm*rCST.D2R)
        sinp, cosp = sin(pm*rCST.D2R), cos(pm*rCST.D2R)
        rmoon=rm*np.array([cosp*cosl,cose*cosp*sinl-sine*sinp,sine*cosp*sinl+cose*sinp])
    return rsun, rmoon
    
def sunmoonpos(tutc,erpv,rsun=False,rmoon=False,gmst=False):
    tut = timeadd(tutc,erpv[2])
    rs, rm = sunmoonpos_eci(tut, rsun, rmoon)
    U, gmst_ = eci2ecef(tutc,erpv)
    if rsun:
        rsun = U@rs
    if rmoon:
        rmoon = U@rm
    if gmst:
        gmst = gmst_
    return rsun, rmoon, gmst

def satantoff(time, rs, sat, nav):
    pcv = searchpcv(sat,time,nav.pcvs)
    dant = np.zeros(3)
    erpv = np.zeros(5)
    rsun, _, _ = sunmoonpos(gpst2utc(time),erpv,True)
    r = -rs
    ez = r/np.linalg.norm(r)
    r = rsun-rs
    es = r/np.linalg.norm(r)
    r = np.cross(ez,es)
    ey = r/np.linalg.norm(r)
    ex = np.cross(ey,ez)

    sys,_ = sat2prn(sat)
    if sys==uGNSS.GPS or sys==uGNSS.QZS:
        freq = [rCST.FREQ1,rCST.FREQ2]
    #elif sys==uGNSS.GLO:
        #freq = [sat2freq(sat,CODE_L1C,nav)]
    elif sys==uGNSS.GAL:
        freq = [rCST.FREQ1,rCST.FREQ7]
    elif sys==uGNSS.BDS:
        freq = [rCST.FREQ1_BDS,rCST.FREQ2_BDS]
    elif sys==uGNSS.IRN:
        freq = [rCST.FREQ5,rCST.FREQ9]
        
    den = freq[0]**2-freq[1]**2
    c1 = freq[0]**2/den
    c2 = -freq[1]**2/den
    
    for i in range(3):
        dant1 = pcv.off[0][0]*ex[i]+pcv.off[0][1]*ey[i]+pcv.off[0][2]*ez[i]
        dant2 = pcv.off[1][0]*ex[i]+pcv.off[1][1]*ey[i]+pcv.off[1][2]*ez[i]    
        dant[i] = c1*dant1+c2*dant2

    return dant


class bias_t():
    def __init__(self,sat:int,tst:gtime_t,ted:gtime_t,type1,code1,
                 type2=0,code2=0,bias=0.0,std=0.0,svn=0):
        self.sat  = sat
        self.tst  = tst
        self.ted  = ted
        self.type1 = type1
        self.code1 = code1
        self.type2 = type2
        self.code2 = code2        
        self.bias  = bias
        self.std   = std
        self.svn   = svn

class biasdec():
    def __init__(self):
        self.gnss_tbl = {'G': uGNSS.GPS, 'E': uGNSS.GAL, 'J': uGNSS.QZS}
        self.sig_tbl = {'1C': rSIG.L1C, '1X': rSIG.L1X, '1W': rSIG.L1W,
                        '2W': rSIG.L2W, '2L': rSIG.L2L, '2X': rSIG.L2X,
                        '5Q': rSIG.L5Q, '5X': rSIG.L5X, '7Q': rSIG.L7Q,
                        '7X': rSIG.L7X}  
        self.dcb = []

    def sig2code(self,sig):
        sig_t = {'C':0,'L':1}
        if sig[0] in sig_t:
            type_ = sig_t[sig[0]]
        else:
            type_ = -1
        if sig[1:3] in self.sig_tbl:
            code = self.sig_tbl[sig[1:3]]
        else:
            code = -1
        return type_, code
            
    def doy2time(self,ep):
        """ calculate time from doy """
        year = int(ep[0])
        doy  = int(ep[1])
        sec  = int(ep[2])
        if year == 0 and doy == 0 and sec == 0: # undef
            year = 3000
        days = (year-1970)*365+(year-1969)//4+doy-1
        return gtime_t(days*86400+sec)

    def getdcb(self, sat, time, code):
        bias, std = 0, 0
        for dcb in self.dcb:
            if dcb.sat==sat and dcb.code2 == code and \
                timediff(time,dcb.tst)>=0.0 and \
                timediff(time,dcb.ted)<0.0:
                bias = dcb.bias
                std  = dcb.std
                bcode= dcb.code1
                break
        return bias, std, bcode
        

    def parse(self,fname):
        with open(fname,"r") as fh:
            status = False
            for line in fh:
                if line[0] == '*':
                    continue
                if '+BIAS/SOLUTION' in line:
                    status = True   
                elif '-BIAS/SOLUTION' in line:
                    status = False
                if status and line[0:5] == ' DSB ': 
                    # Differential Signal Bias
                    svn   = int(line[7:10])
                    prn   = line[11:14]
                    sat   = id2sat(prn)
                    #sname = line[15:24]
                    obs1  = line[25:29]
                    obs2  = line[30:34]
                    type1,code1 = self.sig2code(obs1)
                    type2,code2 = self.sig2code(obs2)                
                    # year:doy:sec
                    ep1   = [int(line[35:39]),int(line[40:43]),int(line[44:49])]
                    ep2   = [int(line[50:54]),int(line[55:58]),int(line[59:64])]
                    tst = self.doy2time(ep1)
                    ted = self.doy2time(ep2)
                    unit  = line[65:69]
                    if type1!=type2:
                        print("format error: type1!=type2")
                        return -1
                    if (type1==0 and unit[0:2]!='ns') or (type1==1 and unit[0:3]!='cyc'):
                        print("format error: inconsistent dimension")
                        return -1
                    
                    bias  = float(line[70:91])
                    std   = float(line[92:103])
                    if len(line)>=137:
                        slope = float(line[104:125])
                        std_s = float(line[126:137])
                    print("{:3d} {:3d} {:s} {:s} {:f}".format(svn,sat,obs1,obs2,bias))        
                    dcb = bias_t(sat,tst,ted,type1,code1,type2,code2,bias,std,svn)
                    self.dcb.append(dcb)

                
                
                                
                
    
    
             
                

if __name__ == '__main__':

    bdir='C:/work/RTKLIB-dev/test/utest/'
    obsfile = bdir+"../data/sp3/igs15904.sp3"
    clkfile = bdir+"../data/sp3/igs15904.clk"
    atxfile = bdir+"../../data/igs05.atx"
    dcbfile = bdir+"../../data/dcb/DLR0MGXFIN_20223310000_01L_07D_DCB.BSX"
    
    time =  epoch2time([2010, 7, 1, 0, 0, 0])
    sat = 3
    #rs, dts, var = sp.peph2pos(time, sat)
    
    rnx = rnxdec()
    nav = Nav()
    
    if False:
        sp = peph()
        nav = sp.parse_sp3(obsfile, nav)
        nav = rnx.decode_clk(clkfile, nav)
        nav.pcvs = readpcv(atxfile)

        n = 10
        rs = np.zeros((n,6))
        dts = np.zeros((n,2))
        for k in range(n):
            t = timeadd(time,30*k)
            rs[k,:], dts[k,:], var = sp.peph2pos(t, sat, nav)

    if False:
        rs, dts, var = sp.peph2pos(time, sat, nav)
        off = satantoff(time, rs[0:3], sat, nav)
        
    if False:
        erpv = np.zeros(5)
        ep1 = [2010,12,31,8,9,10]
        rs = [70842740307.0837,115293403265.153,-57704700666.9715]
        rm = [350588081.147922,29854134.6432052,-136870369.169738]
        rsun, rmoon, _ = sunmoonpos(epoch2time(ep1),erpv,True,True)
        assert np.all(abs((rsun-rs)/rsun)<0.03)
        assert np.all(abs((rmoon-rm)/rmoon)<0.03)
        

    if True:
        time =  epoch2time([2022,12,31, 0, 0, 0])
        sat = 3
    
        bd = biasdec()    
        bd.parse(dcbfile)
        bias,std,bcode = bd.getdcb(sat,time,rSIG.L1W)
        assert bias == -1.2715
        assert std == 0.0058
        assert bcode == rSIG.L1C
    
    #satantoff(time, rs, sat, nav)
    
        