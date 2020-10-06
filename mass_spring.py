import taichi as ti
import random
import math
ti.init(arch=ti.cuda, device_memory_fraction=0.3)

WIDTH = 512
HEIGHT = 512
Radius = 10

MASS = 20
GRAVITY = ti.Vector([0, -400])
dt = 0.005 #0.01667
G = 10000

K = 10000
LENGTH = 4*Radius

particleMax = 256
particleNum = ti.field(ti.i32, shape=())
x = ti.Vector(2, dt=ti.f32, shape=(2, 2))  # shape=(particleMax*3, 2))
v = ti.Vector(2, dt=ti.f32, shape=(2, 2))  # shape=(particleMax*3, 2))
f = ti.Vector(2, dt=ti.f32, shape=(2, 2))  # shape=(particleMax*3, 2))

@ti.kernel
def step():
    for i, j in x:
        f[i, j] = GRAVITY*MASS
        for di, dj in ti.ndrange((-1,2), (-1,2)):
            if(abs(di+dj) == 1):
                ii = i + di
                jj = j + dj
                if ii >= 0 and jj >= 0 and ii <= 1 and jj <= 1:
                    f[i, j] += K*((x[ii, jj]-x[i, j]).norm() - LENGTH) * (x[ii, jj]-x[i, j]).normalized()

        v[i, j] += f[i, j]*dt/MASS

        if x[i, j][1] < Radius:
            v[i, j][1] = 0
            x[i, j][1] = Radius

        x[i, j] += v[i, j]*dt


@ti.kernel
def newParticle(pos_x: ti.f32, pos_y: ti.f32):
    for a, b in ti.ndrange(2, 2):
        x[a, b] = ti.Vector([256 + a*LENGTH*0.5 - b*LENGTH*0.5, 256 + a*LENGTH*0.5 + b*LENGTH*0.5])


gui = ti.GUI('mass_spring', res=(WIDTH, HEIGHT), background_color=0xffffff)
newParticle(256, 256)
while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        # elif e.key == ti.GUI.LMB:
        #     if particleNum[None] < particleMax:
        #         newParticle(e.pos[0]*WIDTH, e.pos[1]*HEIGHT)
        elif e.key == 'r':
            particleNum[None] = 0
    step()
    for i, j in ti.ndrange(2, 2):
        gui.circle((x[i, j][0]/WIDTH, x[i, j][1]/HEIGHT),radius=Radius,color=0x068587)
    gui.show()
