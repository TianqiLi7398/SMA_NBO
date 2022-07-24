from matplotlib.patches import Ellipse, Rectangle, FancyArrowPatch
from scipy.stats.distributions import chi2
import numpy as np

class effects:

	@staticmethod
	def make_ellipse(mean, cov, gating_size, edge_color):
		"""Support function for scatter_ellipse."""

		v, w = np.linalg.eigh(cov)
		u = w[0] / np.linalg.norm(w[0])
		if np.isclose(u[0,0], 0):
			angle = np.sign(u[0,1]) * 0.5 * np.pi
		else:
			angle = np.arctan(u[0, 1]/u[0,0])
		angle = 180 * angle / np.pi # convert to degrees
		v = 2 * np.sqrt(v * gating_size) #get size corresponding to level
		return Ellipse(mean[:2], v[0], v[1], 180 + angle, facecolor='none',
					edgecolor=edge_color,
					#ls='dashed',  #for debugging
					lw=1.5)

	@staticmethod
	def left_bot_point(x, y, theta, h, w):
		a0 = np.arctan(h/w)
		l = np.sqrt(h**2 + w**2)
		theta += a0 
		x -= 0.5 * l * np.cos(theta)
		y -= 0.5 * l * np.sin(theta)
		return x, y

	@staticmethod
	def euclidean_dist(p1, p2):
		return np.sqrt((p2[0] -p1[0])**2 + (p2[1] - p1[1])**2)

	@staticmethod
	def make_rectangle(x, y, theta, para, color=None, linestyle='-'):
		# create rectangle of sensors object in matplotlib animation
		h = para["shape"][1][1]
		w = para["shape"][1][0]
		# bot left position
		x0, y0 = effects.left_bot_point(x, y, theta, h, w)
		if color == None:
			return Rectangle((x0, y0), 
						w, h,
						angle = theta * 180 / np.pi,
						fc ='none',  
						ec =para['color'], 
						lw = 1,
						linestyle = linestyle)
		else:
			return Rectangle((x0, y0), 
						w, h,
						angle = theta * 180 / np.pi,
						fc ='none',  
						ec =color, 
						lw = 1.2,
						linestyle = linestyle)

	@staticmethod
	def make_fancy_arrow(p1, p2):
		e = FancyArrowPatch((p1[0], p1[1]), (p2[0], p2[1]),
							arrowstyle='<-',
							linewidth=2,
							linestyle='dotted',
							color='k')
		return e

	@staticmethod
	def vector(pos, r):

		x = pos[0]
		y = pos[1]
		theta = pos[2]
		x += r * np.cos(theta)
		y += r * np.sin(theta)
		return [x, y]