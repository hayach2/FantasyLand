import numpy as np
from PIL import Image
from transform import normalized
from node import Node
from mesh import Mesh
from transform import rotate, translate


class TextureGround(Mesh, Node):

	def __init__(self, shader, heightmapFile):
		heightmapTexture = np.asarray(Image.open(heightmapFile).convert('RGB'))
		vertices, textureCoordinates, normals, indices = self.create_attributes(heightmapTexture.shape[0],
		                                                                        heightmapTexture)
		super().__init__(shader, [vertices, textureCoordinates, normals], indices)

	def create_attributes(self, size, heightmapTexture):
		vertices = []
		normals = []
		textureCoordinates = []

		for i in range(0, size):
			for j in range(0, size):
				vertices.append(
					[(j / (size - 1)) * 1000, self.get_height(i, j, heightmapTexture), (i / (size - 1)) * 1000])
				normals.append(self.calculateNormal(j, i, heightmapTexture))
				textureCoordinates.append([j / (size - 1), i / (size - 1)])

		vertices = np.array(vertices)
		normals = np.array(normals)
		textureCoordinates = np.array(textureCoordinates)

		indices = []
		for gz in range(0, size - 1):
			for gx in range(0, size - 1):
				top_left = (gz * size) + gx
				top_right = top_left + 1
				bottom_left = ((gz + 1) * size) + gx
				bottom_right = bottom_left + 1
				indices.append([top_left, bottom_left, top_right, top_right, bottom_left, bottom_right])

		indices = np.array(indices)

		return vertices, textureCoordinates, normals, indices

	def calculateNormal(self, x, z, hmap_image):
		return normalized(np.array([self.get_height(x - 1, z, hmap_image) - self.get_height(x + 1, z, hmap_image), 2.0,
		                            self.get_height(x, z - 1, hmap_image) - self.get_height(x, z + 1, hmap_image)]))

	def get_height(self, x, z, image):
		if x < 0 or x >= image.shape[0] or z < 0 or z >= image.shape[0]:
			return 0
		height = image[x, z, 0]
		return (height / 256) * 100

	def key_handler(self, key):
		return


def groundNode(viewer, shader):
	mainNode = Node(transform=translate(-300, 0, -300) @ rotate((1, 0, 0), 0))
	mainNode.add(TextureGround(shader, heightmapFile="../resources/hmap.png"))
	viewer.add(mainNode)
