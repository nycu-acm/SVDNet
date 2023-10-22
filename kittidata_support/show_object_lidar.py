import numpy as np
import mayavi.mlab as mlab

def draw_lidar(pc, color=None, fig=None, bgcolor=(0,0,0), pts_scale=1, pts_mode='point', pts_color=None):
    if fig is None: fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1600, 1000))
    if color is None: color = pc[:,3]*2
    mlab.points3d(pc[:,0], pc[:,1], pc[:,2], color, color=pts_color, mode=pts_mode, colormap = 'gnuplot', scale_factor=pts_scale, figure=fig)
    return fig

path = "C:/Users/kk/Desktop/40.62.bin"
pc_velo = np.fromfile(path, dtype=np.float32).reshape(-1, 4)

fig = mlab.figure(figure=None, bgcolor=(1,1,1), fgcolor=None, engine=None, size=(200, 150))
draw_lidar(pc_velo, fig=fig)
mlab.show(1)