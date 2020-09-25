import taichi as ti
import random
import math
ti.init(arch=ti.gpu)

WIDTH = 512
HEIGHT = 512
Radius = 10

MASS = 20
GRAVITY = ti.Vector([0, -400])
dt = 0.01667

particleMax = 256
particleNum = ti.field(ti.i32, shape=())
x = ti.Vector(2, dt=ti.f32, shape=particleMax)
v = ti.Vector(2, dt=ti.f32, shape=particleMax)
f = ti.Vector(2, dt=ti.f32, shape=particleMax)


@ti.kernel
def step():
    for i in range(particleNum[None]):
        f[i] = GRAVITY*MASS
        if v[i].norm() > 0:  f[i] += v[i].normalized()*(-3*v[i].norm())
        # for j in range(particleNum[None]):
        #     if i != j and (x[i] - x[j]).norm() < Radius*2:
        #         f[i] += (MASS*v[j] - MASS*v[i])/dt
        # v[i] += f[i]*dt/MASS
        #边界反弹
        v[i] += f[i]*dt/MASS
        if x[i][0] <= Radius or x[i][0] >= WIDTH:
            v[i][0] *= -1
        if x[i][1] <= Radius:
            v[i][1] *= -0.8
            x[i][1] = Radius

        x[i] += v[i]*dt


@ti.kernel
def newParticle(pos_x: ti.f32, pos_y: ti.f32):
    x[particleNum[None]] = ti.Vector([pos_x, pos_y])
    v[particleNum[None]] = ti.Vector([pos_x, pos_y])
    particleNum[None] += 1


gui = ti.GUI('Free Fall', res=(WIDTH, HEIGHT), background_color=0xffffff)
newParticle(256, 256)
while True:
    for e in gui.get_events(ti.GUI.PRESS):
        if e.key in [ti.GUI.ESCAPE, ti.GUI.EXIT]:
            exit()
        elif e.key == ti.GUI.LMB:
            if particleNum[None] < particleMax:
                newParticle(e.pos[0]*WIDTH, e.pos[1]*HEIGHT)
        elif e.key == 'r':
            particleNum[None] = 0
    step()
    for i in range(particleNum[None]):
        gui.circle((x[i][0]/WIDTH, x[i][1]/HEIGHT),
                   color= 0xff00ff, radius=Radius)
    gui.show()
