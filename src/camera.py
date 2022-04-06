import copy
import numpy as np
import glfw
from PIL import Image

from transform import normalized


# https://learnopengl.com/Getting-started/Camera
class Camera:
	def __init__(self):
		# Camera position
		self.cameraPos = np.array((0.0, 0.0, 3.0))
		# Camera direction
		self.cameraTarget = np.array((0.0, 0.0, 0.0))
		self.cameraDirection = normalized((self.cameraPos - self.cameraTarget))
		# Right axis
		self.up = np.array((0.0, 1.0, 0.0))
		self.cameraRight = normalized(np.cross(self.up, self.cameraDirection))
		# Up axis
		self.cameraUp = np.cross(self.cameraDirection, self.cameraRight)

		# Walk Around
		self.cameraFront = np.array((0.0, 0.0, 1.0))

		self.sensitivity = 0.01

		self.hmap_file = "../resources/hmap.png"
		self.heightmapTexture = np.asarray(Image.open(self.hmap_file).convert('RGB'))

	def processInput(self, window, deltaTime):
		self.cameraPos[1] = self.cameraPositionXZ() + 3

		cameraFrontCopy = copy.deepcopy(self.cameraFront)
		cameraFrontCopy[1] = 0

		if glfw.get_key(window=window, key=glfw.KEY_RIGHT_SHIFT):
			self.sensitivity = 0.1
			camera_speed = 50 * deltaTime
		else:
			self.sensitivity = 0.01
			camera_speed = 20 * deltaTime
		
		if glfw.get_key(window=window, key=glfw.KEY_W):
			self.cameraPos += camera_speed * cameraFrontCopy
		if glfw.get_key(window=window, key=glfw.KEY_S):
			self.cameraPos -= camera_speed * cameraFrontCopy
		if glfw.get_key(window=window, key=glfw.KEY_A):
			self.cameraPos -= normalized(np.cross(cameraFrontCopy, self.up)) * camera_speed
		if glfw.get_key(window=window, key=glfw.KEY_D):
			self.cameraPos += normalized(np.cross(cameraFrontCopy, self.up)) * camera_speed
		if glfw.get_key(window=window, key=glfw.KEY_LEFT):
			if self.cameraFront[2] >= 0 and self.cameraFront[0] >= 0:
				self.cameraFront[2] -= self.sensitivity
				self.cameraFront[0] += self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
			if self.cameraFront[2] < 0 and self.cameraFront[0] >= 0:
				self.cameraFront[2] -= self.sensitivity
				self.cameraFront[0] -= self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
			if self.cameraFront[2] < 0 and self.cameraFront[0] < 0:
				self.cameraFront[2] += self.sensitivity
				self.cameraFront[0] -= self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
			if self.cameraFront[2] >= 0 and self.cameraFront[0] < 0:
				self.cameraFront[2] += self.sensitivity
				self.cameraFront[0] += self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
		if glfw.get_key(window=window, key=glfw.KEY_RIGHT):
			if self.cameraFront[2] >= 0 and self.cameraFront[0] >= 0:
				self.cameraFront[2] += self.sensitivity
				self.cameraFront[0] -= self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
			if self.cameraFront[2] < 0 and self.cameraFront[0] >= 0:
				self.cameraFront[2] += self.sensitivity
				self.cameraFront[0] += self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
			if self.cameraFront[2] < 0 and self.cameraFront[0] < 0:
				self.cameraFront[2] -= self.sensitivity
				self.cameraFront[0] += self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
			if self.cameraFront[2] >= 0 and self.cameraFront[0] < 0:
				self.cameraFront[2] -= self.sensitivity
				self.cameraFront[0] -= self.sensitivity
				self.cameraFront = normalized(self.cameraFront)
		if glfw.get_key(window=window, key=glfw.KEY_UP):
			if self.cameraFront[1] < 1:
				self.cameraFront[1] += self.sensitivity
		if glfw.get_key(window=window, key=glfw.KEY_DOWN):
			if self.cameraFront[1] > 0:
				self.cameraFront[1] -= self.sensitivity

	def cameraPositionXZ(self):
		x = int(((self.cameraPos[2] + 300) / 1000) * self.heightmapTexture.shape[0])
		z = int(((self.cameraPos[0] + 300) / 1000) * self.heightmapTexture.shape[0])
		if 0 < x < self.heightmapTexture.shape[0] or 0 < z < self.heightmapTexture.shape[0]:
			return (self.heightmapTexture[x, z, 0] / 256) * 100
		else:
			return 0

	def get_cameraPos(self):
		return self.cameraPos

	def get_cameraFront(self):
		return self.cameraFront

	def get_cameraUp(self):
		return self.up
