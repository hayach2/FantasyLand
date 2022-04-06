from itertools import cycle
import glfw
import OpenGL.GL as GL

from transform import identity, lookat, perspective
from node import Node
from camera import Camera


# ------------  Viewer class & window management ------------------------------
class Viewer(Node):
    """ GLFW viewer window, with classic initialization & graphics loop """

    def __init__(self, width=640, height=480):
        super().__init__()

        self.width = width
        self.height = height
        self.camera = Camera()
        self.lastFrame = 0.0

        # version hints: create GL window with >= OpenGL 3.3 and core profile
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL.GL_TRUE)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.RESIZABLE, True)
        self.win = glfw.create_window(width, height, 'Viewer', None, None)

        # make win's OpenGL context current; no OpenGL calls can happen before
        glfw.make_context_current(self.win)

        # register event handlers
        glfw.set_key_callback(self.win, self.on_key)
        glfw.set_window_size_callback(self.win, GL.glViewport(0, 0, *glfw.get_framebuffer_size(self.win)))

        # initialize GL by setting viewport and default render characteristics
        GL.glClearColor(0.1, 0.1, 0.1, 0.1)
        # GL.glClearColor(0.52, 0.8, 0.91, 0.2)
        GL.glEnable(GL.GL_CULL_FACE)  # backface culling enabled (TP2)
        GL.glEnable(GL.GL_DEPTH_TEST)  # depth test now enabled (TP2)

    def run(self):
        """ Main render loop for this OpenGL window """
        while not glfw.window_should_close(self.win):
            # clear draw buffer and depth buffer (<-TP2)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

            current_frame = glfw.get_time()
            delta_time = current_frame - self.lastFrame
            self.lastFrame = current_frame

            # Update the view matrix with camera orientation
            view = lookat(eye=self.camera.get_cameraPos(),
                          target=self.camera.get_cameraPos() + self.camera.get_cameraFront(),
                          up=self.camera.get_cameraUp())
            projection = perspective(fovy=45, aspect=(self.width / self.height), near=0.1, far=500.0)
            self.draw(projection, view, identity())

            # flush render commands, and swap draw buffers
            glfw.swap_buffers(self.win)

            # Poll for and process events
            glfw.poll_events()

            self.camera.processInput(window=self.win, deltaTime=delta_time)

    def on_key(self, _win, key, _scancode, action, _mods):
        """ 'Q' or 'Escape' quits """
        if action == glfw.PRESS or action == glfw.REPEAT:
            if key == glfw.KEY_ESCAPE or key == glfw.KEY_Q:
                glfw.set_window_should_close(self.win, True)
            if key == glfw.KEY_R:
                GL.glPolygonMode(GL.GL_FRONT_AND_BACK, next(self.fill_modes))
            if key == glfw.KEY_SPACE:
                glfw.set_time(0)

            # call Node.key_handler which calls key_handlers for all drawables
            self.key_handler(key)
