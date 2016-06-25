import numpy as np
import scipy as sp
import scipy.special as sps 
import reduce_integrals as ri

# import math

def shell_pl(epsnot,sindex,rmin,rmax,radarr,c=1,ff=1e-3,epsatrmin=0):

##############################################################
### Written by Charles Romero. IRAM.
###
### PURPOSE: Integrate a power law function (similiar to emissivity) 
###          along the z-axis (i.e. line of sight). This performs the
###          integration analytically.
###
### HISTORY:
### 23.06.2013 - CR: Written, not tested.
###
##############################################################
### INPUTS:
#
# EPSNOT    - The normalization factor. The default behavior is for
#             this to be defined at RMAX, the outer edge of a sphere
#             or shell. If you integrate to infinity, then this should
#             be defined at RMIN. And of course, RMIN=0, and RMAX as
#             infinity provides no scale on which to define EPSNOT.
#             See the optional variable EPSATRMIN.
# SINDEX    - "Spectral Index". That is, the power law 
#             (without the minus sign) that the "emissivity"
#             follows within your bin. This program only works for
#             SINDX > 1 (strictly greater than 1!)
# RMIN      - Minimum radius for your bin. Can be 0.
# RMAX      - Maximum radius for your bin. If you wish to set this
#             to infinity, then set it to a negative value.
#
### -- NOTE -- If RMIN = 0 and RMAX < 0, then this program will return 0.
#
# RADARR    - A radial array of projected radii (same units as RMIN
#             and RMAX) for which projected values will be calculated.
#             If the innermost value is zero, its value, in the scaled
#             radius array will be set to FF.
# [C=1]     - The scaling axis for an ellipse along the line of sight.
#             The default 
# [FF=1e-3] - Fudge Factor. If the inner
# [EPSATRMIN] - Set this to a value greater than 0 if you want EPSNOT to be
#               defined at RMIN. This automatically happens if RMAX<0
#
##############################################################
### OUTPUTS:
#
# PLINT     - PLINT is the integration along the z-axis (line of sight) for
#             an ellipsoid (a sphere) where the "emissivity" is governed by
#             a power law. The units are thus given as the units on EPSNOT
#             times the units on RADARR (and therefore RMIN and RMAX).
#
#             It is then dependent on you to make the appropriate
#             conversions to the units you would like.
# 
##############################################################

  if rmin < 0:
    print 'found rmin < 0; setting rmin equal to 0'
    rmin = 0

  if rmax>0 and rmax<rmin:
    print 'You made a mistake: rmin > rmax'

  if rmax < 0:
      if rmin == 0:
          scase=3
      else:
          scase=2
          epsatrmin=1
  else:
      if rmin == 0:
          scase=0
      else:
          scase=1
          epsatrmin=1

  rrmm = (radarr==np.amin(radarr))
  if (radarr[rrmm] == 0) and (sindex > 0):
    radarr[rrmm]=ff

###############################

  shellcase = {0: plsphere, # You are integrating from r=0 to R (finite)
               1: plshell,  # You are integrating from r=R_1 to R_2 (finite)
               2: plsphole, # You are integrating from r=R (finite, >0) to infinity
               3: plinfty,  # You are integrating from r=0 to infinity
           }

  p = sindex/2.0 # e(r) = e_0 * (r^2)^(-p) for this notation / program

  myintegration = shellcase[scase](p,rmin,rmax,radarr)

  if epsatrmin > 0:
      epsnorm=epsnot*(rmax/rmin)**(sindex)
  else:
      epsnorm=epsnot

  prefactors=epsnorm*c
  answer = myintegration*prefactors
  return answer

##############################

def plsphere(p,rmin,rmax,radarr):
    c1 = radarr<=rmax              # condition 1
    c2 = radarr>rmax               # condition 2
    sir=(radarr[c1]/rmax)          # scaled radii
    isni=((2.0*p==np.floor(2.0*p)) and (p<=1))
    if isni:
      tmax=np.arctan(np.sqrt(1.0 - sir**2)/sir)
      plint=ri.myredcosine(tmax,2.0*p-2.0)*(sir**(1.0-2.0*p))*2.0
    else:
      cbf=(sps.gamma(p-0.5)*np.sqrt(np.pi))/sps.gamma(p) # complete beta function
      ibir=ri.myrincbeta(sir**2,p-0.5,0.5) # incomplete beta function
      plint=(sir**(1.0-2.0*p))*(1.0-ibir)*cbf

#    import pdb; pdb.set_trace()
    myres=radarr*0          # Just make my array (unecessary?)
    myres[c1]=plint         # Define values for R < RMIN
    return myres*rmax               # The results we want

def plshell(p,rmin,rmax,radarr):
    c1 = radarr<=rmax               # condition 1
    c2 = radarr[c1]<rmin            # condition 2
    sir=(radarr[c1]/rmin)           # scaled inner radii
    sor=(radarr[c1]/rmax)           # scaled outer radii
    isni=((2.0*p==np.floor(2.0*p)) and (p<=1))
    myres=radarr*0                  # Just make my array (unecessary?)
    if isni:
      tmax=np.arctan(np.sqrt(1.0 - sor**2)/sor)
      tmin=np.arctan(np.sqrt(1.0 - sir[c2]**2)/sir[c2])
      plint=ri.myredcosine(tmax,2.0*p-2.0)
      plint[c2]-=ri.myredcosine(tmin,2.0*p-2.0)
      myres[c1]=plint*(sor**(1.0-2.0*p))*2.0
      
    else:
      cbf=(sps.gamma(p-0.5)*np.sqrt(np.pi))/sps.gamma(p) # complete beta function
      ibir=ri.myrincbeta(sir**2,p-0.5,0.5) # Inc. Beta for inn. rad.
      ibor=ri.myrincbeta(sor**2,p-0.5,0.5) # Inc. Beta for out. rad.
      plinn=(sir**(1.0-2.0*p))       # Power law term for inner radii
      #    plout=(sor**(1.0-p))      # Power law term for outer radii
      myres[c1]=plinn*(1.0-ibor)*cbf      # Define values for the enclosed circle
      myres[c2]=plinn[c2]*(ibir[c2]-ibor[c2])*cbf # Correct the values for the 
                                        # innermost circle
    return myres*rmin               # The results we want

#    tosub=plsphere(p,0,rmin,radarr)
#    return myres-tosub*(rmin/rmax)^p
#    myres[c1]=plout*(1.0-ibor) - plinn*(1.0-ibir)

def plsphole(p,rmin,rmax,radarr):
    c1 = radarr<rmin               # condition 1
    c2 = radarr>=rmin              # condition 2
    sr=(radarr/rmin)               # scaled radii
    cbf=(sps.gamma(p-0.5)*np.sqrt(np.pi))/sps.gamma(p) # complete beta function
    ibor=ri.myrincbeta(sr[c1]**2,p-0.5,0.5) # Inc. Beta for out. rad.
    plt=(sr**(1.0-2.0*p))          # Power law term
    myres=radarr*0                 # Just make my array (unecessary?)
    myres[c1]=plt[c1]*ibor*cbf     # Define values for R < RMIN
    myres[c2]=plt[c2]*cbf          # Define values for R > RMIN
    return myres*rmin

def plinfty(p,rmin,rmax,radarr):
    return 0       # Scale invariant. Right. Fail.

