#This code uses Moehlis low-dimensional model for turbulent shear flows and solve nine coupled Ordinary differential equations(ODEs)
#It also plots time-evolution of nine-coefficients (a1, a2,...a9), instantaneous velocity plots (in-plane quiver and contours in out of plane) and streamwise mean velocity profile (x-averaged u_x vs y)
#References: Jeff Moehlis et. al. 2004, New J. Phys. 6, 56 and P.A. Srinivasan, Phys. Rev. Fluids, 4, 054603(2019)
#This code is written by Pawandeep Kaur on 13 April 2023
 

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from pylab import*

#Function defining nine ODEs
def Moehlis(y, t, p):
	a1, a2, a3, a4, a5, a6, a7, a8, a9 = y
	p = [a, b, g, Re]
	k_ag = np.sqrt(a**2 + g**2)
	k_ab = np.sqrt(a**2 + b**2)
	k_bg = np.sqrt(b**2 + g**2)
	k_abg = np.sqrt(a**2 + b**2 + g**2)
	da1dt = (b**2)/Re - (b**2/Re)*a1 - np.sqrt(3/2)*(b*g/k_abg)*a6*a8 + np.sqrt(3/2)*(b*g/k_bg)*a2*a3
	da2dt = -(4*b**2/3+g**2)*(a2/Re) + ((5*np.sqrt(2)*g**2)/(3*np.sqrt(3)*k_ag))*a4*a6 - (g**2/(np.sqrt(6)*k_ag))*a5*a7 - ((a*b*g)/(np.sqrt(6)*k_ag*k_abg))*a5*a8 - np.sqrt(3/2)*((b*g)/(k_bg))*a1*a3 - np.sqrt(3/2)*((b*g)/(k_bg))*a3*a9
	da3dt = -(k_bg**2/Re)*a3 + ((2.*a*b*g)/(np.sqrt(6)*k_ag*k_bg))*(a4*a7+a5*a6) + ((b**2*(3*a**2+g**2)-3*g**2*k_ag**2)/(np.sqrt(6)*k_ag*k_bg*k_abg))*a4*a8
	da4dt = -((3*a**2+4*b**2)/(3*Re))*a4 - (a/np.sqrt(6))*a1*a5 - ((10*a**2)/(3*np.sqrt(6)*k_ag))*a2*a6 - np.sqrt(3/2)*((a*b*g)/(k_ag*k_bg))*a3*a7 - np.sqrt(3/2)*((a**2*b**2)/(k_ag*k_bg*k_abg))*a3*a8 -(a/np.sqrt(6))*a5*a9
	da5dt = -(k_ab**2/Re)*a5 + (a/np.sqrt(6))*a1*a4 + (a**2/(np.sqrt(6)*k_ag))*a2*a7 - (a*b*g/(np.sqrt(6)*k_ag*k_abg))*a2*a8 + (a/np.sqrt(6))*a4*a9 + ((2*a*b*g)/(np.sqrt(6)*k_ag*k_bg))*a3*a6
	da6dt = -((3*a**2+4*b**2+3*g**2)/(3*Re))*a6 + (a/np.sqrt(6))*a1*a7 + (np.sqrt(3/2)*b*g/(k_abg))*a1*a8 + (10*(a**2-g**2)/(3*np.sqrt(6)*k_ag))*a2*a4 - (2*np.sqrt(2/3)*a*b*g/(k_ag*k_bg))*a3*a5 + (a/np.sqrt(6))*a7*a9 + (np.sqrt(3/2)*b*g/k_abg)*a8*a9
	da7dt = -(k_abg**2/Re)*a7 - a*(a1*a6+a6*a9)/np.sqrt(6) + ((g**2-a**2)/(np.sqrt(6)*k_ag))*a2*a5 + ((a*b*g)/(np.sqrt(6)*k_ag*k_bg))*a3*a4
	da8dt = -(k_abg**2/Re)*a8 + ((2*a*b*g)/(np.sqrt(6)*k_ag*k_abg))*a2*a5 + (g**2*(3*a**2-b**2+3*g**2)/(np.sqrt(6)*k_ag*k_bg*k_abg))*a3*a4
	da9dt = -(9*b**2/Re)*a9 + (np.sqrt(3/2)*b*g/k_bg)*a2*a3 - (np.sqrt(3/2)*b*g/k_abg)*a6*a8
	func = [da1dt, da2dt, da3dt, da4dt, da5dt, da6dt, da7dt, da8dt, da9dt]
	return func

#Input parameters
Re = 400
Lx =4*np.pi
Ly = 2
Lz = 2*np.pi
a = 2*np.pi/Lx #alpha
b = np.pi/2.	#beta
g = 2*np.pi/Lz #gamma
p = [a, b, g, Re]

#Initial Conditions
y0 = [1, 0.07066, -0.07076, 0.01, 0, 0, 0, 0, 0]
#y0 = [0.1, 0.01, 0., 0, 0, -0.08, 0, 0, -0.1]

time = np.linspace(0, 4000, 4001)    #time array
sol = odeint(Moehlis, y0, time, args = (p,))  # Gives solution of nine ODEs

y = np.linspace(-1, 1, 50)
x = np.linspace(0, Lx, 60) 
z = np.linspace(0, Lz, 60)


u1x = np.zeros((len(y),len(z)))
u2x = np.zeros((len(y),len(z)))
u3y = np.zeros((len(y),len(z)))
u3z = np.zeros((len(y),len(z)))
u4z = np.zeros((len(y),len(z)))
u5z = np.zeros((len(y),len(z)))
u6x = np.zeros((len(y),len(z)))
u6z = np.zeros((len(y),len(z)))
u7x = np.zeros((len(y),len(z)))
u7z = np.zeros((len(y),len(z))) 
u8x = np.zeros((len(y),len(z)))
u8y = np.zeros((len(y),len(z)))
u8z = np.zeros((len(y),len(z))) 
u9x = np.zeros((len(y),len(z)))


#The following part of code stores x, y and z components of velocities ( U_X(time, y, z), U_Y(time, y, z), Y_Z(time, y, z) ) averaged over x 
for k in range(len(z)):
	for j in range(len(y)):
		#n = j + len(y)*k
		for i in range(len(x)):
			u1x[j,k] += (np.sqrt(2)*np.sin(np.pi*y[j]/2.))
			u2x[j,k] += (4/np.sqrt(3))*(np.cos(np.pi*y[j]/2))**2*np.cos(g*z[k])
			u3y[j,k] += (2./np.sqrt(4*g**2+np.pi**2))*2*g*np.cos(np.pi*y[j]/2)*np.cos(g*z[k])
			u3z[j,k] += (2./np.sqrt(4*g**2+np.pi**2))*np.pi*np.sin(np.pi*y[j]/2)*np.sin(g*z[k])
			u4z[j,k] += (4/np.sqrt(3))*np.cos(a*x[i])*(np.cos(np.pi*y[j]/2))**2
			u5z[j,k] += 2*np.sin(a*x[i])*np.sin(np.pi*y[j]/2)
			u6x[j,k] += (4*np.sqrt(2)/np.sqrt(3*(a**2+g**2)))*(-g)*np.cos(a*x[i])*(np.cos(np.pi*y[j]/2))**2*np.sin(g*z[k])
			u6z[j,k] += (4*np.sqrt(2)/np.sqrt(3*(a**2+g**2)))*a*np.sin(a*x[i])*(np.cos(np.pi*y[j]/2))**2*np.cos(g*z[k])
			u7x[j,k] += (2*np.sqrt(2)/np.sqrt(a**2+g**2))*g*np.sin(a*x[i])*np.sin(np.pi*y[j]/2)*np.sin(g*z[k])
			u7z[j,k] += (2*np.sqrt(2)/np.sqrt(a**2+g**2))*a*np.cos(a*x[i])*np.sin(np.pi*y[j]/2)*np.cos(g*z[k])
			u8x[j,k] += ( 2*np.sqrt(2)/(np.sqrt((a**2+g**2)*(4*a**2+4*g**2+np.pi**2))) )*np.pi*a*np.sin(a*x[i])*np.sin(np.pi*y[j]/2)*np.sin(g*z[k])
			u8y[j,k] += ( 2*np.sqrt(2)/(np.sqrt((a**2+g**2)*(4*a**2+4*g**2+np.pi**2))) )*2*(a**2+g**2)*np.cos(a*x[i])*np.cos(np.pi*y[j]/2)*np.sin(g*z[k])
			u8z[j,k] += ( 2*np.sqrt(2)/(np.sqrt((a**2+g**2)*(4*a**2+4*g**2+np.pi**2))) )*(-np.pi*g)*np.cos(a*x[i])*np.sin(np.pi*y[j]/2)*np.cos(g*z[k])
			u9x[j,k] += np.sqrt(2)*np.sin(3*np.pi*y[j]/2)
		u1x[j,k] /= len(x)
		u2x[j,k] /= len(x)
		u3y[j,k] /= len(x)
		u3z[j,k] /= len(x)
		u4z[j,k] /= len(x)
		u5z[j,k] /= len(x)
		u6x[j,k] /= len(x)
		u6z[j,k] /= len(x)
		u7x[j,k] /= len(x)
		u7z[j,k] /= len(x)
		u8x[j,k] /= len(x)
		u8y[j,k] /= len(x)
		u8z[j,k] /= len(x)
		u9x[j,k] /= len(x)


#print(len(u3y))
#print(len(u5z))


#print(U1)
dimV = (len(time), len(y), len(z))
U_X = np.zeros(dimV)
U_Y = np.zeros(dimV)
U_Z = np.zeros(dimV)

#print(np.shape(U_X))
#print(np.shape(U_Y))
#print(np.shape(U_Z))


for i in range(4001):
	#print(time[i])
	for k in range(len(z)):
		for j in range(len(y)):
			t1 = time[i]
			U_X[i, j, k] = sol[i,0]*u1x[j,k] + sol[i,1]*u2x[j,k] + sol[i,5]*u6x[j,k] + sol[i,6]*u7x[j,k] + sol[i,7]*u8x[j,k] + sol[i,8]*u9x[j,k] 
			U_Y[i, j, k] = sol[i,2]*u3y[j,k] + sol[i,7]*u8y[j,k]
			U_Z[i, j, k] = sol[i,2]*u3z[j,k] + sol[i,3]*u4z[j,k] + sol[i,4]*u5z[j,k] + sol[i,5]*u6z[j,k] + sol[i,6]*u7z[j,k] + sol[i,7]*u8z[j,k]
 

#print(U_X.shape)
print("%.2f\n" %t1)

#The following portion of code stores nine coefficients as a function of time
f = open("Time_evolution_of_Moehlis_coefficients_a4_0p01_PA_Srinivasan.data", "w")
#f = open("1.data", "w")

for i in range(4001):
	f.write("%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n" %(time[i], sol[i, 0], sol[i, 1], sol[i, 2], sol[i, 3], sol[i, 4], sol[i, 5], sol[i, 6], sol[i, 7], sol[i, 8]))

f.close()

#To plot time-evolution of nine coefficients(a1, a2,...., a9) of Moehlis model
fig1 = plt.figure(figsize = (8, 8))

#fig1.suptitle(r"$a_1 = 0.1, a_2 = 0.01, a_6 = -0.08, a_9 = -0.1$")
fig1.suptitle(r"$a_1 = 1, a_2 = 0.7066, a_3 = -0.7076, a_4 = 0.01, a_5=..=a_9=0$")
plt.subplot(5, 2, 1)
plt.plot(time, sol[:, 0], '-r', label = r"$a_1$")
plt.legend()
plt.subplot(5, 2, 2)
plt.plot(time, sol[:, 1], '-b', label = r"$a_2$")
plt.legend()
plt.subplot(5, 2, 3)
plt.plot(time, sol[:, 2], '-g', label = r"$a_3$")
plt.legend()
plt.subplot(5, 2, 4)
plt.plot(time, sol[:, 3], '-y', label = r"$a_4$")
plt.legend()
plt.subplot(5, 2, 5)
plt.plot(time, sol[:, 4], '-k', label = r"$a_5$")
plt.legend()
plt.subplot(5, 2, 6)
plt.plot(time, sol[:, 5], '-m', label = r"$a_6$")
plt.legend()
plt.subplot(5, 2, 7)
plt.plot(time, sol[:, 6], '-c', label = r"$a_7$")
plt.legend()
plt.subplot(5, 2, 8)
plt.plot(time, sol[:, 7], '--', color = 'r', label = r"$a_8$")
plt.legend()
plt.subplot(5, 2, 9)
plt.plot(time, sol[:, 8], '--', color = 'b', label = r"$a_9$")

plt.legend()
plt.savefig("Temporal_Evolution_of_nine_Coefficients_Moehlis_Lx_4pi_Ly_2pi_Re_400_a4_0p01.png")
plt.show()


t = 20 #time at which instantaneous plots will be obtained
#Uncomment the following to plot instantaneous in-plane velocity quiver plot and out-of-plane contour plot at particular time value
'''fig2, ax = plt.subplots()
Y, Z = np.meshgrid(y, z)
plt.xlabel("$z$")
plt.ylabel("$y$")

plt.contourf(Z, Y, np.transpose(U_X[t, :, :]), levels = 50, cmap = 'jet') 
plt.colorbar()
plt.quiver(Z, Y, np.transpose(U_Z[t, :, :]), np.transpose(U_Y[t, :, :]), color = 'k')
plt.title(r"$t = {0}$".format(t), fontsize = 15)
#plt.savefig("Instantaneous_velocities_t_{0}_Moehlis_Lx_4pi_Lz_2pi_Re_400_a1_1000.png".format(t))
plt.show()
'''

#Uncomment the following commands to plot instantaneous mean velocity(x-averaged) along x-direction(u_x) as a function of y(bounded direction) at a fix time value
'''fig3 = plt.figure()
U_mean = np.mean(U_X[t,:,:], -1)
plt.xlabel(r"$y$")
plt.ylabel(r"$\bar{u}_x$")
plt.title(r"$t = {0}$".format(t), fontsize = 15)
plt.plot(y, U_mean, '-r')
#plt.savefig("Mean_profile_t_{0}_Moehlis_Lx_4pi_Lz_2pi_Re_400_a1_1000.png".format(t))
plt.show()
'''



