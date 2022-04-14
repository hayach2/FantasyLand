import numpy as np
import glfw
import OpenGL.GL as GL
from mesh import Mesh
from node import Node
from keyframe import TransformKeyFrames
from transform import identity

# -------------- Linear Blend Skinning : TP7 ---------------------------------
MAX_VERTEX_BONES = 4
MAX_BONES = 128


class SkinnedMesh(Mesh):
	"""class of skinned mesh nodes in scene graph """

	def __init__(self, shader, attribs, bone_nodes, bone_offsets, index=None):
		super().__init__(shader, attribs, index)

		self.bone_nodes = bone_nodes
		self.bone_offsets = np.array(bone_offsets, np.float32)

	def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
		""" skinning object draw method """
		GL.glUseProgram(self.shader.glid)

		world_transforms = [node.world_transform for node in self.bone_nodes]
		bone_matrix = world_transforms @ self.bone_offsets
		loc = GL.glGetUniformLocation(self.shader.glid, 'bone_matrix')
		GL.glUniformMatrix4fv(loc, len(self.bone_nodes), True, bone_matrix)

		super().draw(projection, view, model)


# -------- Skinning Control for Keyframing Skinning Mesh Bone Transforms ------
class SkinningControlNode(Node):
	""" Place node with transform keys above a controlled subtree """

	def __init__(self, *keys, transform=identity(), delay=None):
		super().__init__(transform=transform)
		self.keyframes = TransformKeyFrames(*keys) if keys[0] else None
		self.world_transform = identity()
		self.time = glfw.get_time()

		self.delay = delay

	def draw(self, projection, view, model):
		""" When redraw requested, interpolate our node transform from keys """
		self.time = glfw.get_time()
		if self.keyframes:
			if self.delay is not None:
				self.time = glfw.get_time() % self.delay
			self.transform = self.keyframes.value(self.time)

		self.world_transform = model @ self.transform

		super().draw(projection, view, model)
