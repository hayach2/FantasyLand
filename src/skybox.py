import OpenGL.GL as GL
import numpy as np

from cubemap import Cubemap
from shader import Shader
from vertexarray import VertexArray
from transform import rotate, scale

# Drawing the cube faces, each face takes up 2 triangles
CUBE_VERTS = [np.array((
	(-1.0, 1.0, -1.0),
	(-1.0, -1.0, -1.0),
	(1.0, -1.0, -1.0),
	(1.0, -1.0, -1.0),
	(1.0, 1.0, -1.0),
	(-1.0, 1.0, -1.0),

	(-1.0, -1.0, 1.0),
	(-1.0, -1.0, -1.0),
	(-1.0, 1.0, -1.0),
	(-1.0, 1.0, -1.0),
	(-1.0, 1.0, 1.0),
	(-1.0, -1.0, 1.0),

	(1.0, -1.0, -1.0),
	(1.0, -1.0, 1.0),
	(1.0, 1.0, 1.0),
	(1.0, 1.0, 1.0),
	(1.0, 1.0, -1.0),
	(1.0, -1.0, -1.0),

	(-1.0, -1.0, 1.0),
	(-1.0, 1.0, 1.0),
	(1.0, 1.0, 1.0),
	(1.0, 1.0, 1.0),
	(1.0, -1.0, 1.0),
	(-1.0, -1.0, 1.0),

	(-1.0, 1.0, -1.0),
	(1.0, 1.0, -1.0),
	(1.0, 1.0, 1.0),
	(1.0, 1.0, 1.0),
	(-1.0, 1.0, 1.0),
	(-1.0, 1.0, -1.0),

	(-1.0, -1.0, -1.0),
	(-1.0, -1.0, 1.0),
	(1.0, -1.0, -1.0),
	(1.0, -1.0, -1.0),
	(-1.0, -1.0, 1.0),
	(1.0, -1.0, 1.0)), 'f')]


class Skybox:
	def __init__(self, files):
		self.shader = Shader('shaders/skybox.vert', 'shaders/skybox.frag')
		self.vertex_array = VertexArray(CUBE_VERTS)
		self.cubemap = Cubemap(files)

	def draw(self, projection, view, model, color_shader=None, win=None, **param):
		""" Draw object """
		GL.glDepthFunc(GL.GL_LEQUAL)
		GL.glUseProgram(self.shader.glid)

		# projection geometry
		loc = GL.glGetUniformLocation(self.shader.glid, 'modelviewprojection')
		GL.glUniformMatrix4fv(loc, 1, True, projection @ view @ scale(200, 200, 200) @ np.identity(4, 'f'))

		# texture access setups
		loc = GL.glGetUniformLocation(self.shader.glid, 'skybox')
		GL.glActiveTexture(GL.GL_TEXTURE0)
		GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, self.cubemap.glid)

		GL.glUniform1i(loc, 0)
		self.vertex_array.execute(GL.GL_TRIANGLES)

		GL.glDepthFunc(GL.GL_LESS)

		GL.glBindTexture(GL.GL_TEXTURE_CUBE_MAP, 0)
		GL.glUseProgram(0)
