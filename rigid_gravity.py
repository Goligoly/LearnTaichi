import taichi as ti
import random
ti.init(arch=ti.gpu)

WIDTH = 512
HEIGHT = 512
Radius = 5

MASS = 20
GRAVITY = ti.Vector([0, -200])
dt = 0.01667

particleMax = 256
particleNum = ti.field(ti.i32, shape=())
x = ti.Vector(2, dt=ti.f32, shape=particleMax)
v = ti.Vector(2, dt=ti.f32, shape=particleMax)


@ti.kernel
def step():
    for i in range(particleNum[None]):
        v[i] += GRAVITY*dt
        if x[i][0] <= Radius or x[i][0] >= WIDTH:
            v[i][0] = -v[i][0]
        if x[i][1] <= Radius or x[i][1] >= HEIGHT:
            v[i][1] = -v[i][1]
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
            newParticle(e.pos[0]*WIDTH, e.pos[1]*HEIGHT)
    step()
    for i in range(particleNum[None]):
        gui.circle((x[i][0]/WIDTH, x[i][1]/HEIGHT),
                   color=0xFF6EB4, radius=Radius)
    gui.show()
