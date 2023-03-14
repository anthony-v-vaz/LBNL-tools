# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 13:37:06 2017

@author: lo5959
"""

from PIL import Image
import numpy as np
import math
#import matplotlib
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
import sys



#focus = "Bulli"
focus = "Mini"


spat_cal = 0.13 # um/px
size_jet = 10. # diameter in um
pp_delays_cyl =  [0,10,20,50,100]
size_jet_cyl = [6.,15.9,24.6,37.5,48.7]
std_size_jet_cyl = [0.,0.5,1.2,4.6,7.1]
size_jet_cyl_vert = 50.
size_jet_plan = [16.]
std_size_jet_plan = [0.]
offset = 0. # offset of jet position

im = np.asarray(Image.open('hdr_norm.tif'.format(focus)))
im_max = np.max(im)
im_min = np.min(im)
print(im_max)
print(im_min)
print(np.where(im==im_max))
print(im.shape)

im_norm = np.divide(im, im_max)*3.
print(np.max(im_norm))
print(np.where(im_norm==np.max(im_norm)))
print(np.min(im_norm))

#im_norm_neg = np.subtract(im_norm,1.)
#print(np.max(im_norm_neg))
#print(np.min(im_norm_neg))
im_exp_notnorm = np.power(10.,im_norm)
im_exp_full = np.divide(im_exp_notnorm,np.max(im_exp_notnorm))
print("np.max(im_exp_full)")
print(np.max(im_exp_full))
print("np.min(im_exp_full)")
print(np.min(im_exp_full))
#np.savetxt("hdr_focus.csv", im_exp_full, delimiter=",")

full_transm = np.sum(im_exp_full)
x_edges = np.arange(1,im_exp_full.shape[0]+1,1.)*0.13
y_edges = np.arange(1,im_exp_full.shape[1]+1,1.)*0.13
print(len(x_edges))
print(len(y_edges))
print(x_edges[0],x_edges[-1])
A,P = np.meshgrid(x_edges,y_edges)

vmin=np.min(im_exp_full)
vmax=np.max(im_exp_full)

plt.figure()
plt.pcolormesh(A,P,im_exp_full.T,cmap='CMRmap_r',norm=LogNorm(vmin=vmin,vmax=vmax))
plt.axes().set_aspect('equal')
plt.xlabel('x [um]')
plt.ylabel('y [um]')
cbar = plt.colorbar()
cbar.set_label('signal / arb. u.', fontsize=22)
cbar.ax.tick_params(labelsize=16)
plt.tight_layout()
#plt.savefig('{}_hdr_focus.png'.format(focus), format='png')

plt.show()

sys.exit()

ind_min = 0.
ind_max = 0.

tl_hdr = []

#for s,size_jet in enumerate(size_jet_cyl):
#    print(pp_delays_cyl[s])
for i, item in enumerate(x_edges.tolist()):
    if item < len(x_edges)/2.*spat_cal-size_jet/2.-offset:
        if x_edges.tolist()[i+1] > len(x_edges)/2.*spat_cal-size_jet/2.-offset:
            ind_min = i
    if item < len(x_edges)/2.*spat_cal+size_jet/2.-offset:
        if x_edges.tolist()[i+1] > len(x_edges)/2.*spat_cal+size_jet/2.-offset:
            ind_max = i
print('a')
for p,ppd in enumerate(pp_delays_cyl):
    im_exp = im_exp_full
    print('pp delay = '+str(ppd)+'ps')
    for r,row in enumerate(im_exp.T):
        
#        if r == round(len(y_edges)/2,0):

        if -size_jet_cyl_vert/2./spat_cal < r-int(len(y_edges)/2.) < size_jet_cyl_vert/2./spat_cal: 
            for i, item in enumerate(row):
#                print(-size_jet_cyl_vert/2./spat_cal)
#                print(r-int(len(y_edges)/2))
#                print((size_jet_cyl_vert/2.)**2.)
#                print((y_edges[r]-int(len(y_edges)/2.)*spat_cal)**2.)
                l = -math.sqrt((size_jet_cyl_vert/2.)**2.-((r-int(len(y_edges)/2.))*spat_cal)**2.)*size_jet_cyl[p]/size_jet_cyl_vert
#                print('l')
#                print(l)
                u = -l
#                print('u')
#                print(u)
                if l < (i-int(len(row)/2.))*spat_cal < u:
#                    print(item)
#                    print(im_exp[i,r])
                    im_exp[i,r] = 0.
#                        print('x_edges.tolist()[i]')
#                        print(x_edges.tolist()[i]-len(x_edges)/2*spat_cal)
               
    plt.figure()
    plt.pcolormesh(A,P,im_exp.T,cmap='CMRmap_r',norm=LogNorm(vmin=vmin,vmax=vmax))
    plt.axes().set_aspect('equal')
    plt.xlabel('x [um]')
    plt.ylabel('y [um]')
    cbar = plt.colorbar()
    cbar.set_label('signal / arb. u.', fontsize=22)
    cbar.ax.tick_params(labelsize=16)
    
    plt.show()

    # array without part in the middle where light is blocked by the jet
    im_exp_wjet = np.concatenate((im_exp.T[:,:ind_min],im_exp.T[:,ind_max:]),axis=1).T
    ratio = np.sum(im_exp_wjet)/full_transm
    print('ratio=')
    print(ratio)
    tl_hdr.append(ratio)


    # plot truncated arrays:
    A_min,P_min = np.meshgrid(x_edges[:ind_min],y_edges)
    
    plt.figure()
    plt.pcolormesh(A_min,P_min,im_exp.T[:,:ind_min],cmap='CMRmap_r',norm=LogNorm(vmin=vmin,vmax=vmax))
    plt.axes().set_aspect('equal')
    cbar = plt.colorbar()
    cbar.set_label('particle number / MeV / sr', fontsize=22)
    cbar.ax.tick_params(labelsize=16)
    
    plt.show()

#A_max,P_max = np.meshgrid(x_edges[ind_max:],y_edges)
#
#plt.figure()
#plt.pcolormesh(A_max,P_max,im_exp.T[:,ind_max:],cmap='CMRmap_r',norm=LogNorm(vmin=vmin,vmax=vmax))
#plt.axes().set_aspect('equal')
#cbar = plt.colorbar()
#cbar.set_label('particle number / MeV / sr', fontsize=22)
#cbar.ax.tick_params(labelsize=16)
#
#plt.show()

#slize_hor = im_exp.T[[int(im_exp.shape[0]/2)],:]
#print(slize_hor.shape)

#sys.exit()

# compare to experimentally measured transmitted light values:
path_tl_exp = '../../../../2016/H2_Draco/Auswertung/macor_imaging/'
tl_plan = np.load(path_tl_exp+'tl_plan.npy',mmap_mode=None, allow_pickle=True, fix_imports=True)
tl_plan_std = np.load(path_tl_exp+'tl_plan_std.npy',mmap_mode=None, allow_pickle=True, fix_imports=True)
tl_cyl = np.load(path_tl_exp+'tl_pp_exp.npy',mmap_mode=None, allow_pickle=True, fix_imports=True)
tl_cyl_std = np.load(path_tl_exp+'tl_pp_exp_std.npy',mmap_mode=None, allow_pickle=True, fix_imports=True)

plt.figure()
plt.plot(pp_delays_cyl, tl_hdr,'o',color='orange',label='cylindrical, hdr')
plt.plot(0.,tl_hdr[1],'d',color='orange',label='planar, hdr')
plt.errorbar(pp_delays_cyl,tl_cyl,yerr=tl_cyl_std,fmt='o',color='blue',label='cylindrical, macor')
plt.errorbar(0.,tl_plan,yerr=tl_plan_std,fmt='d',color='blue',label='planar, macor')
plt.xlabel('pre-pulse delay / ps')
plt.ylabel('transmitted light / arb. u.')
plt.legend()
plt.show()
sys.exit()

#for (x,y), value in np.ndenumerate(im_exp):
#    if  int(im_exp.shape[2])<= x



#im_exp = np.power(10.,im)

#print(np.max(im_exp))
#print(np.min(im_exp))




dyn_im = im_max - im_min
dyn_real = 1000.



