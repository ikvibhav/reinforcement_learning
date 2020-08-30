from gym.envs.box2d import car_racing
from pyglet.window import key
import numpy as np

a = np.array([0.0, 0.0, 0.0])

def key_press(k, mod):
    global restart
    if k == 0xff0d: restart = True
    if k == key.LEFT:  a[0] = -1.0
    if k == key.RIGHT: a[0] = +1.0
    if k == key.UP:    a[1] = +1.0
    if k == key.DOWN:  a[2] = +0.8  # set 1.0 for wheels to block to zero rotation

def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0: a[0] = 0
    if k == key.RIGHT and a[0] == +1.0: a[0] = 0
    if k == key.UP:    a[1] = 0
    if k == key.DOWN:  a[2] = 0

env = car_racing.CarRacing()
env.render()
record_video = False
if record_video:
    env.monitor.start('/tmp/video-test', force=True)
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
while True:
    env.reset()
    total_reward = 0.0
    steps = 0
    restart = False
    while True:
        s, r, done, info = env.step(a)
        total_reward += r
        if steps % 200 == 0 or done:
            print("\naction " + str(["{:+0.2f}".format(x) for x in a]))
            print("step {} total_reward {:+0.2f}".format(steps, total_reward))
            # import matplotlib.pyplot as plt
            # plt.imshow(s)
            # plt.savefig("test.jpeg")
        steps += 1
        if not record_video:  # Faster, but you can as well call env.render() every time to play full window.
            env.render()
        if done or restart: break
env.close()