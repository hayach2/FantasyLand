#!/usr/bin/env python3
"""
Python OpenGL Medieval Project.
"""
# Python built-in modules
import glfw  # lean window system wrapper for OpenGL

# External, non built-in modules
from viewer import Viewer
from skybox import Skybox
from shader import Shader
from textureground import groundNode
from core import add_characters, add_animations


def main():
    """ create a window, add scene objects, then run rendering loop """

    # Define all the shaders
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

    add_characters(viewer, shader=skinning_shader)
    # add_animations(viewer, shader=phong_shader)
    groundNode(viewer, shader=terrain_shader)


    viewer.add(Skybox(field))

    message = """
    Welcome to our Fantasy World!

    Press SPACEBAR to reset all animations.

    And finally, press ESC or Q to exit the game.
    """
    print(message)

    # start rendering loop
    viewer.run()


if __name__ == '__main__':
    glfw.init()  # initialize window system glfw
    main()  # main function keeps variables locally scoped
    glfw.terminate()  # destroy all glfw windows and GL contexts
