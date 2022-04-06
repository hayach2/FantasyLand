import OpenGL.GL as GL
import numpy as np
import glfw
# import assimpcy

# import os  # os function, i.e. checking file status

from vertexarray import VertexArray
from node import Node
# import config
from color import Color
color = Color()

# ------------  Mesh is a core drawable, can be basis for most objects --------
class Mesh:
    """ Basic mesh class with attributes passed as constructor arguments """

    def __init__(self, shader, attributes, index=None):
        self.shader = shader
        self.names = ['view', 'projection', 'model']
        self.loc = {n: GL.glGetUniformLocation(shader.glid, n) for n in self.names}
        self.vertex_array = VertexArray(attributes, index)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        GL.glUniformMatrix4fv(self.loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(self.loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(self.loc['model'], 1, True, model)

        # draw triangle as GL_TRIANGLE vertex array, draw array call
        self.vertex_array.execute(primitives)


# -------------- Texture based Phong rendered Mesh class -------------------------

class TexturedPhongMesh(Node):
    def __init__(self, shader, tex, attributes, faces,
                 light_dir=None, k_a=(1, 1, 1), k_d=(1, 1, 0), k_s=(1, 1, 0),
                 s=64.):
        # super().__init__(shader, tex, attributes, faces)
        super().__init__()
        # setup texture and upload it to GPU
        self.texture = tex
        self.vertex_array = VertexArray(attributes=attributes, index=faces)
        self.shader = shader

        self.k_a = k_a
        self.k_d = k_d
        self.k_s = k_s
        self.s = s
        # ----------------

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # projection geometry
        names = ['view', 'projection', 'model',
                 'diffuseMap', 'k_a', 'k_d', 'k_s', 's',
                 'color', 'w_camera_position']
        loc = {n: GL.glGetUniformLocation(self.shader.glid, n) for n in names}

        # Iterate over all the light sources and send (to shader) their properties.
        for i in range(0, color.num_light_src):
            light_pos_loc = GL.glGetUniformLocation(self.shader.glid, 'light_position[%d]' % i)
            GL.glUniform3fv(light_pos_loc, 1, color.light_pos[i])

            atten_loc = GL.glGetUniformLocation(self.shader.glid, 'atten_factor[%d]' % i)
            GL.glUniform3fv(atten_loc, 1, color.get_atten()[i])

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        GL.glUniform3fv(loc['k_a'], 1, self.k_a)
        GL.glUniform3fv(loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(loc['k_s'], 1, self.k_s)
        GL.glUniform1f(loc['s'], max(self.s, 0.001))
        GL.glUniform3fv(loc['color'], 1, color.get_color())

        # world camera position for Phong illumination specular component
        w_camera_position = np.linalg.inv(view)[:, 3]
        GL.glUniform3fv(loc['w_camera_position'], 1, w_camera_position)

        # ----------------
        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc['diffuseMap'], 0)
        self.vertex_array.execute(primitives)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)


class TexturedPhongMeshSkinned(Node):
    def __init__(self, shader, tex, attributes, faces, bone_nodes, bone_offsets,
                 k_a=(1, 1, 1), k_d=(1, 1, 0), k_s=(1, 1, 0), s=64.):
        super().__init__()

        # setup texture and upload it to GPU
        self.texture = tex
        self.vertex_array = VertexArray(attributes=attributes, index=faces)
        self.shader = shader

        self.k_a = k_a
        self.k_d = k_d
        self.k_s = k_s
        self.s = s
        # ----------------
        self.bone_nodes = bone_nodes
        self.bone_offsets = np.array(bone_offsets, np.float32)

    def draw(self, projection, view, model, primitives=GL.GL_TRIANGLES):
        GL.glUseProgram(self.shader.glid)

        # projection geometry
        names = ['view', 'projection', 'model', 'diffuseMap', 'k_a', 'k_d', 'k_s', 's',
                 'color', 'w_camera_position']
        loc = {n: GL.glGetUniformLocation(self.shader.glid, n) for n in names}

        GL.glUniformMatrix4fv(loc['view'], 1, True, view)
        GL.glUniformMatrix4fv(loc['projection'], 1, True, projection)
        GL.glUniformMatrix4fv(loc['model'], 1, True, model)

        GL.glUniform3fv(loc['k_a'], 1, self.k_a)
        GL.glUniform3fv(loc['k_d'], 1, self.k_d)
        GL.glUniform3fv(loc['k_s'], 1, self.k_s)
        GL.glUniform1f(loc['s'], max(self.s, 0.001))
        GL.glUniform3fv(loc['color'], 1, color.get_color())

        # bone world transform matrices need to be passed for skinning
        for bone_id, node in enumerate(self.bone_nodes):
            bone_matrix = node.world_transform @ self.bone_offsets[bone_id]

            bone_loc = GL.glGetUniformLocation(self.shader.glid, 'bone_matrix[%d]' % bone_id)
            GL.glUniformMatrix4fv(bone_loc, len(self.bone_nodes), True, bone_matrix)

        # Iterate over all the light sources and send (to shader) their properties.
        for i in range(0, color.num_light_src):
            light_pos_loc = GL.glGetUniformLocation(self.shader.glid, 'light_position[%d]' % i)
            GL.glUniform3fv(light_pos_loc, 1, color.light_pos[i])

            atten_loc = GL.glGetUniformLocation(self.shader.glid, 'atten_factor[%d]' % i)
            GL.glUniform3fv(atten_loc, 1, color.get_atten()[i])

        # world camera position for Phong illumination specular component
        w_camera_position = np.linalg.inv(view)[:, 3]
        GL.glUniform3fv(loc['w_camera_position'], 1, w_camera_position)

        # ----------------
        # texture access setups
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture.glid)
        GL.glUniform1i(loc['diffuseMap'], 0)
        self.vertex_array.execute(primitives)

        # leave clean state for easier debugging
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glUseProgram(0)



# def load_textured_phong_mesh_skinned(file, shader, tex_file, k_a, k_d, k_s, s, delay=None):
#     try:
#         pp = assimpcy.aiPostProcessSteps
#         flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
#         scene = assimpcy.aiImportFile(file, flags)
#     except assimpcy.all.AssimpError as exception:
#         print('ERROR loading', file + ': ', exception.args[0].decode())
#         return []
#     # print("Materials: ", scene.mNumMaterials)
#     # Note: embedded textures not supported at the moment
#     path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
#     for mat in scene.mMaterials:
#         if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
#             name = os.path.basename(mat.properties['TEXTURE_BASE'])
#             # search texture in file's whole subdir since path often screwed up
#             paths = os.walk(path, followlinks=True)
#             found = [os.path.join(d, f) for d, _, n in paths for f in n
#                     if name.startswith(f) or f.startswith(name)]
#             assert found, 'Cannot find texture %s in %s subtree' % (name, path)
#             tex_file = found[0]
#         if tex_file:
#             mat.properties['diffuse_map'] = Texture(tex_file=tex_file)

#     # ----- load animations
#     def conv(assimp_keys, ticks_per_second):
#         """ Conversion from assimp key struct to our dict representation """
#         return {key.mTime / ticks_per_second: key.mValue for key in assimp_keys}

#     # load first animation in scene file (could be a loop over all animations)
#     transform_keyframes = {}
#     if scene.mAnimations:
#         anim = scene.mAnimations[0]
#         for channel in anim.mChannels:
#             # for each animation bone, store TRS dict with {times: transforms}
#             transform_keyframes[channel.mNodeName] = (
#                 conv(channel.mPositionKeys, anim.mTicksPerSecond),
#                 conv(channel.mRotationKeys, anim.mTicksPerSecond),
#                 conv(channel.mScalingKeys, anim.mTicksPerSecond)
#             )

#     # ---- prepare scene graph nodes
#     # create SkinningControlNode for each assimp node.
#     # node creation needs to happen first as SkinnedMeshes store an array of
#     # these nodes that represent their bone transforms
#     nodes = {}  # nodes name -> node lookup
#     nodes_per_mesh_id = [[] for _ in scene.mMeshes]  # nodes holding a mesh_id

#     def make_nodes(assimp_node):
#         """ Recursively builds nodes for our graph, matching assimp nodes """
#         trs_keyframes = transform_keyframes.get(assimp_node.mName, (None,))
#         skin_node = SkinningControlNode(*trs_keyframes,
#                                         transform=assimp_node.mTransformation, delay=delay)
#         nodes[assimp_node.mName] = skin_node
#         for mesh_index in assimp_node.mMeshes:
#             nodes_per_mesh_id[mesh_index].append(skin_node)
#         skin_node.add(*(make_nodes(child) for child in assimp_node.mChildren))
#         return skin_node

#     root_node = make_nodes(scene.mRootNode)

#     # ---- create SkinnedMesh objects
#     for mesh_id, mesh in enumerate(scene.mMeshes):
#         # -- skinned mesh: weights given per bone => convert per vertex for GPU
#         # first, populate an array with MAX_BONES entries per vertex
#         v_bone = np.array([[(0, 0)] * MAX_BONES] * mesh.mNumVertices,
#                         dtype=[('weight', 'f4'), ('id', 'u4')])
#         for bone_id, bone in enumerate(mesh.mBones[:MAX_BONES]):
#             for entry in bone.mWeights:  # weight,id pairs necessary for sorting
#                 v_bone[entry.mVertexId][bone_id] = (entry.mWeight, bone_id)

#         v_bone.sort(order='weight')  # sort rows, high weights last
#         v_bone = v_bone[:, -MAX_VERTEX_BONES:]  # limit bone size, keep highest

#         # prepare bone lookup array & offset matrix, indexed by bone index (id)
#         bone_nodes = [nodes[bone.mName] for bone in mesh.mBones]
#         bone_offsets = [bone.mOffsetMatrix for bone in mesh.mBones]

#         # Initialize mat for phong and texture
#         # mat = scene.mMaterials[mesh.mMaterialIndex].properties
#         # assert mat['diffuse_map'], "Trying to map using a textureless material"

#     # meshes = []
#     for mesh in scene.mMeshes:
#         mat = scene.mMaterials[mesh.mMaterialIndex].properties
#         assert mat['diffuse_map'], "Trying to map using a textureless material"
#         attributes = [mesh.mVertices, mesh.mTextureCoords[0], mesh.mNormals, v_bone['id'], v_bone['weight']]
#         mesh = TexturedPhongMeshSkinned(shader=shader, tex=mat['diffuse_map'], attributes=attributes,
#                                         faces=mesh.mFaces, bone_nodes=bone_nodes, bone_offsets=bone_offsets,
#                                         k_d=k_d, k_a=k_a, k_s=k_s, s=s)

#         for node in nodes_per_mesh_id[mesh_id]:
#             node.add(mesh)

#         nb_triangles = sum((mesh.mNumFaces for mesh in scene.mMeshes))

#     return [root_node]
