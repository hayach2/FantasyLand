#!/usr/bin/env python3

import glfw
from viewer import Viewer
from skybox import Skybox
from shader import Shader
from textureground import groundNode
from core import add_animation, add_objects


def main():
    viewer = Viewer(width=1920, height=1080)
    # -------------------------------------------------
    field = [
        '../resources/skybox/lf.jpg',
        '../resources/skybox/rt.jpg',
        '../resources/skybox/up.jpg',
        '../resources/skybox/dn.jpg',
        '../resources/skybox/ft.jpg',
        '../resources/skybox/bk.jpg']

    phong_shader = Shader("shaders/phong.vert", "shaders/phong.frag")
    skinning_shader = Shader("shaders/skinning.vert", "shaders/skinning.frag")
    terrain_shader = Shader("shaders/ground.vert", "shaders/ground.frag")

    add_animation(viewer, shader=skinning_shader)
    add_objects(viewer, shader=phong_shader)
    groundNode(viewer, shader=terrain_shader)


    viewer.add(Skybox(field))

    print("""
    Hi and welcome to our FantasyLand!
    
    Here's how you can navigate our land:
    
    W: Move forward
    S: Move backwards
    D: Move to the right
    A: Move to the left
    UP arrow: Move camera upwards
    Down arrow: Move camera downwards
    Right arrow: Move camera to the right
    Left arrow: Move camera to the left
    Shift (works with all the buttons): Makes movements of all previous keys faster
    Spacebar to reset all animations
    ESC or Q to exit the land
    
    """)

    viewer.run()


if __name__ == '__main__':
    glfw.init()  # initialize window system glfw
    main()  # main function keeps variables locally scoped
    glfw.terminate()  # destroy all glfw windows and GL contexts
