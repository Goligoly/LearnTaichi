import taichi as ti
ti.init(arch=ti.gpu)

WIDTH = 512
HEIGHT = 512
Radius = 5

MASS = 20
GRAVITY = ti.Vector([0, -200])
dt = 0.01667

x = ti.Vector(2, dt=ti.f32, shape=())
v = ti.Vector(2, dt=ti.f32, shape=())
x[None] = ti.Vector([256, 256])
v[None] = ti.Vector([100, 0])


@ti.kernel
def step():
    v[None] += GRAVITY*dt
    if x[None][0] <= Radius or x[None][0] >= WIDTH:
        v[None][0] = -v[None][0]
    if x[None][1] <= Radius or x[None][1] >= HEIGHT:
        v[None][1] = -v[None][1]
    x[None] += v[None]*dt


gui = ti.GUI('Free Fall', res=(WIDTH, HEIGHT), background_color=0xffffff)
while True:
    step()
    gui.circle((x[None][0]/WIDTH, x[None][1]/HEIGHT), color=0xFF6EB4, radius=Radius)
    gui.show()
