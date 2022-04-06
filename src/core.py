# Python built-in modules
import os  # os function, i.e. checking file status
import glfw
import assimpcy
import numpy as np  # all matrix manipulations & OpenGL args
import random
import math

# External, non built-in modules
from mesh import TexturedPhongMesh, TexturedPhongMeshSkinned
from skinning import SkinningControlNode, MAX_VERTEX_BONES, MAX_BONES
from texture import Texture
from node import Node
from keyframe import KeyFrameControlNode
from animation import Animation
from transform import quaternion, rotate, translate, scale, vec, quaternion_from_axis_angle
from camera import Camera

camera = Camera()
# --------------------------------------------------------
# Loader functions for loading different types of 3D objects
# (multi-textured, single textured, and skeletal-based)
# --------------------------------------------------------

def load_textured_phong_mesh(file, shader, tex_file, k_a, k_d, k_s, s):
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []

    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            mat.properties['diffuse_map'] = Texture(tex_file=tex_file)

    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mTextureCoords[0], mesh.mNormals]
        mesh = TexturedPhongMesh(shader=shader, tex=mat['diffuse_map'], attributes=attributes,
                                 faces=mesh.mFaces,
                                 k_d=k_d, k_a=k_a, k_s=k_s, s=s)
        meshes.append(mesh)

        size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
        # print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


def load_textured_phong_mesh_skinned(file, shader, tex_file, k_a, k_d, k_s, s, delay=None):
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_GenSmoothNormals
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []
    # print("Materials: ", scene.mNumMaterials)
    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for mat in scene.mMaterials:
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            mat.properties['diffuse_map'] = Texture(tex_file=tex_file)

    # ----- load animations
    def conv(assimp_keys, ticks_per_second):
        """ Conversion from assimp key struct to our dict representation """
        return {key.mTime / ticks_per_second: key.mValue for key in assimp_keys}

    # load first animation in scene file (could be a loop over all animations)
    transform_keyframes = {}
    if scene.mAnimations:
        anim = scene.mAnimations[0]
        for channel in anim.mChannels:
            # for each animation bone, store TRS dict with {times: transforms}
            transform_keyframes[channel.mNodeName] = (
                conv(channel.mPositionKeys, anim.mTicksPerSecond),
                conv(channel.mRotationKeys, anim.mTicksPerSecond),
                conv(channel.mScalingKeys, anim.mTicksPerSecond)
            )

    # ---- prepare scene graph nodes
    # create SkinningControlNode for each assimp node.
    # node creation needs to happen first as SkinnedMeshes store an array of
    # these nodes that represent their bone transforms
    nodes = {}  # nodes name -> node lookup
    nodes_per_mesh_id = [[] for _ in scene.mMeshes]  # nodes holding a mesh_id

    def make_nodes(assimp_node):
        """ Recursively builds nodes for our graph, matching assimp nodes """
        trs_keyframes = transform_keyframes.get(assimp_node.mName, (None,))
        skin_node = SkinningControlNode(*trs_keyframes,
                                        transform=assimp_node.mTransformation, delay=delay)
        nodes[assimp_node.mName] = skin_node
        for mesh_index in assimp_node.mMeshes:
            nodes_per_mesh_id[mesh_index].append(skin_node)
        skin_node.add(*(make_nodes(child) for child in assimp_node.mChildren))
        return skin_node

    root_node = make_nodes(scene.mRootNode)

    # ---- create SkinnedMesh objects
    for mesh_id, mesh in enumerate(scene.mMeshes):
        # -- skinned mesh: weights given per bone => convert per vertex for GPU
        # first, populate an array with MAX_BONES entries per vertex
        v_bone = np.array([[(0, 0)] * MAX_BONES] * mesh.mNumVertices,
                          dtype=[('weight', 'f4'), ('id', 'u4')])
        for bone_id, bone in enumerate(mesh.mBones[:MAX_BONES]):
            for entry in bone.mWeights:  # weight,id pairs necessary for sorting
                v_bone[entry.mVertexId][bone_id] = (entry.mWeight, bone_id)

        v_bone.sort(order='weight')  # sort rows, high weights last
        v_bone = v_bone[:, -MAX_VERTEX_BONES:]  # limit bone size, keep highest

        # prepare bone lookup array & offset matrix, indexed by bone index (id)
        bone_nodes = [nodes[bone.mName] for bone in mesh.mBones]
        bone_offsets = [bone.mOffsetMatrix for bone in mesh.mBones]

        # Initialize mat for phong and texture
        # mat = scene.mMaterials[mesh.mMaterialIndex].properties
        # assert mat['diffuse_map'], "Trying to map using a textureless material"

    # meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mTextureCoords[0], mesh.mNormals, v_bone['id'], v_bone['weight']]
        mesh = TexturedPhongMeshSkinned(shader=shader, tex=mat['diffuse_map'], attributes=attributes,
                                        faces=mesh.mFaces, bone_nodes=bone_nodes, bone_offsets=bone_offsets,
                                        k_d=k_d, k_a=k_a, k_s=k_s, s=s)

        for node in nodes_per_mesh_id[mesh_id]:
            node.add(mesh)

        nb_triangles = sum((mesh.mNumFaces for mesh in scene.mMeshes))

    return [root_node]

def multi_load_textured(file, shader, tex_file, k_a, k_d, k_s, s):
    """ load resources from file using assimp, return list of TexturedMesh """
    try:
        pp = assimpcy.aiPostProcessSteps
        flags = pp.aiProcess_Triangulate | pp.aiProcess_FlipUVs
        scene = assimpcy.aiImportFile(file, flags)
    except assimpcy.all.AssimpError as exception:
        print('ERROR loading', file + ': ', exception.args[0].decode())
        return []
    # print("materials: ", scene.mNumMaterials)
    # Note: embedded textures not supported at the moment
    path = os.path.dirname(file) if os.path.dirname(file) != '' else './'
    for index, mat in enumerate(scene.mMaterials):
        if not tex_file and 'TEXTURE_BASE' in mat.properties:  # texture token
            name = os.path.basename(mat.properties['TEXTURE_BASE'])
            # search texture in file's whole subdir since path often screwed up
            paths = os.walk(path, followlinks=True)
            found = [os.path.join(d, f) for d, _, n in paths for f in n
                     if name.startswith(f) or f.startswith(name)]
            assert found, 'Cannot find texture %s in %s subtree' % (name, path)
            tex_file = found[0]
        if tex_file:
            # print("Index: ", index)
            mat.properties['diffuse_map'] = Texture(tex_file=tex_file[index])

    # prepare textured mesh
    meshes = []
    for mesh in scene.mMeshes:
        mat = scene.mMaterials[mesh.mMaterialIndex].properties
        assert mat['diffuse_map'], "Trying to map using a textureless material"
        attributes = [mesh.mVertices, mesh.mTextureCoords[0], mesh.mNormals]
        mesh = TexturedPhongMesh(shader, mat['diffuse_map'], attributes, mesh.mFaces,
                                 k_d=k_d, k_a=k_a, k_s=k_s, s=s)
        meshes.append(mesh)

    size = sum((mesh.mNumFaces for mesh in scene.mMeshes))
    # print('Loaded %s\t(%d meshes, %d faces)' % (file, len(meshes), size))
    return meshes


# --------------------------------------------------------
# Builder functions
# Utility functions calling the loader functions
# --------------------------------------------------------

def add_characters(viewer, shader):
    # Knight Run
    keyframe_knight_node = KeyFrameControlNode(
        translate_keys={
               0.1: vec(0, camera.cameraPositionXZ() + 3, 30),
                10: vec(-50, camera.cameraPositionXZ() + 3, 30)},
        rotate_keys={ 0.1: quaternion_from_axis_angle(axis=(0, 1, 0), degrees=90), 10: quaternion_from_axis_angle(axis=(0, 1, 0), degrees=90),
                    },
        scale_keys={ 0: 0.5,  9.9: 0.5, 10: 0,
                    },
    )
    size = 0.05
    knight_node = Node(
        transform=translate(0, 0, 10) @ scale(size, size, size))
    mesh_list = load_textured_phong_mesh_skinned("./../resources/characters/Rogalic/Rogalic_run.fbx", shader=shader,
                                                 tex_file="./../resources/characters/Rogalic/Texture/Rogalik_texture.psd",
                                                 k_a=(.4, .4, .4),
                                                 k_d=(.6, .6, .6),
                                                 k_s=(.1, .1, .1),
                                                 s=4, delay=1.0
                                                 )

    for mesh in mesh_list:
        print(mesh)
        knight_node.add(mesh)
    keyframe_knight_node.add(knight_node)
    viewer.add(keyframe_knight_node)

    # Knight Attack
    keyframe_knight_node = KeyFrameControlNode(
        translate_keys={ 10: vec(0, camera.cameraPositionXZ() + 3, 30)},
        rotate_keys={10: quaternion_from_axis_angle(axis=(0, 1, 0), degrees=90)},
        scale_keys={ 10: 0, 10.1: 0.5, 17.9: 0.5, 18: 0},
    )
    size = 0.05
    knight_node = Node(
        transform=translate(0, 0, 10) @ scale(size, size, size))
    mesh_list = load_textured_phong_mesh_skinned("./../resources/characters/Rogalic/Rogalic_attack_1.fbx", shader=shader,
                                                 tex_file="./../resources/characters/Rogalic/Texture/Rogalik_texture.psd",
                                                 k_a=(.4, .4, .4),
                                                 k_d=(.6, .6, .6),
                                                 k_s=(.1, .1, .1),
                                                 s=4, delay=1.0
                                                 )

    for mesh in mesh_list:
        print(mesh)
        knight_node.add(mesh)
    keyframe_knight_node.add(knight_node)
    viewer.add(keyframe_knight_node)

    # Knight Victory
    keyframe_knight_node = KeyFrameControlNode(
        translate_keys={ 17: vec(5, camera.cameraPositionXZ() + 3, 30)},
        rotate_keys={0: quaternion_from_axis_angle(axis=(0, 1, 0), degrees=190)},
        scale_keys={ 17.9: 0, 18: 0.5, 25: 0.5},
    )
    size = 0.05
    knight_node = Node(
        transform=translate(0, 0, 10) @ scale(size, size, size))
    mesh_list = load_textured_phong_mesh_skinned("./../resources/characters/Rogalic/Rogalic_victory.fbx", shader=shader,
                                                 tex_file="./../resources/characters/Rogalic/Texture/Rogalik_texture.psd",
                                                 k_a=(.4, .4, .4),
                                                 k_d=(.6, .6, .6),
                                                 k_s=(.1, .1, .1),
                                                 s=4, delay=1.0
                                                 )

    for mesh in mesh_list:
        print(mesh)
        knight_node.add(mesh)
    keyframe_knight_node.add(knight_node)
    viewer.add(keyframe_knight_node)

    # Golem Idle
    keyframe_golem_node = KeyFrameControlNode(
        translate_keys={
               10: vec(15, camera.cameraPositionXZ() + 3, 30)},
        rotate_keys={10: quaternion_from_axis_angle(axis=(0, 1, 0), degrees=-90),
                    },
        scale_keys={ 0: 0.5,  9.9: 0.5, 10: 0},
    )
    size = 0.05
    golem_node = Node(
        transform=translate(0, 0, 10) @ scale(size, size, size))
    mesh_list = load_textured_phong_mesh_skinned("./../resources/characters/Golem/Golem_idle.fbx", shader=shader,
                                                 tex_file="./../resources/characters/Golem/Texture/Golem.psd",
                                                 k_a=(.4, .4, .4),
                                                 k_d=(.6, .6, .6),
                                                 k_s=(.1, .1, .1),
                                                 s=4, delay=1.0
                                                 )

    for mesh in mesh_list:
        golem_node.add(mesh)
    keyframe_golem_node.add(golem_node)
    viewer.add(keyframe_golem_node)

     # Golem Attack
    keyframe_golem_node = KeyFrameControlNode(
        translate_keys={ 10: vec(15, camera.cameraPositionXZ() + 3, 30)},
        rotate_keys={10: quaternion_from_axis_angle(axis=(0, 1, 0), degrees=-90)},
        scale_keys={ 9.9: 0, 10: 0.5, 17.9: 0.5, 18: 0},
    )
    size = 0.05
    golem_node = Node(
        transform=translate(0, 0, 10) @ scale(size, size, size))
    mesh_list = load_textured_phong_mesh_skinned("./../resources/characters/Golem/Golem_attack_1.fbx", shader=shader,
                                                 tex_file="./../resources/characters/Golem/Texture/Golem.psd",
                                                 k_a=(.4, .4, .4),
                                                 k_d=(.6, .6, .6),
                                                 k_s=(.1, .1, .1),
                                                 s=4, delay=1.0
                                                 )

    for mesh in mesh_list:
        golem_node.add(mesh)
    keyframe_golem_node.add(golem_node)
    viewer.add(keyframe_golem_node)

     # Golem Death
    keyframe_golem_node = KeyFrameControlNode(
        translate_keys={ 17: vec(15, camera.cameraPositionXZ() + 3, 30)},
        rotate_keys={17: quaternion_from_axis_angle(axis=(0, 1, 0), degrees=-90)},
        # scale_keys={ 0: 0.5},
        scale_keys={ 17.9: 0, 18: 0.5, 19.1: 0.5, 19.2: 0},
    )
    size = 0.05
    golem_node = Node(
        transform=translate(0, 0, 10) @ scale(size, size, size))
    mesh_list = load_textured_phong_mesh_skinned("./../resources/characters/Golem/Golem_death.fbx", shader=shader,
                                                 tex_file="./../resources/characters/Golem/Texture/Golem.psd",
                                                 k_a=(.4, .4, .4),
                                                 k_d=(.6, .6, .6),
                                                 k_s=(.1, .1, .1),
                                                 s=4, delay=1.0
                                                 )

    for mesh in mesh_list:
        print(mesh)
        golem_node.add(mesh)
    keyframe_golem_node.add(golem_node)
    viewer.add(keyframe_golem_node)

def add_animations(viewer, shader):
    def circular_motion(r=30, x_offset=0, y_offset=0, z_offset=0, direction=0):
        speed = 20
        angle = (glfw.get_time() * speed) % 360

        # Reverse the direction of rotation
        if direction == 1:
            rev_angle = 360 - angle
            angle = rev_angle
        x = x_offset + (r * math.cos(math.radians(angle)))
        y = y_offset + (np.absolute(10 * math.sin(math.radians(angle))))
        z = z_offset + (r * math.sin(math.radians(angle)))
        transformation = translate(x, camera.cameraPositionXZ() + 40, 150) @ rotate((0, 1, 1), 90)  @ rotate((0, 1, 0), 10) @ rotate((1, 0, 1), 130) @ rotate((1, 0, 1), 180)
        return transformation
        
    # Seagull
    radius = int(random.randrange(start=10, stop=100, step=10))
    x_offset = int(random.randrange(start=0, stop=100, step=5))
    y_offset = int(random.randrange(start=0, stop=10, step=2))
    z_offset = int(random.randrange(start=30, stop=50, step=5))
    direction = int(random.randrange(start=0, stop=2, step=1))

    bird_node = Animation(circular_motion,
                                    radius=radius,
                                    x_offset=x_offset,
                                    y_offset=y_offset,
                                    z_offset=z_offset,
                                    direction=direction)
    mesh_list = load_textured_phong_mesh(file="./../resources/Seagull/seagul.FBX", shader=shader,
                                            tex_file="./../resources/Seagull/texture/gull.png",
                                            k_a=(.4, .4, .4),
                                            k_d=(1.2, 1.2, 1.2),
                                            k_s=(.2, .2, .2),
                                            s=4
                                            )
    for mesh in mesh_list:
        bird_node.add(mesh)
    viewer.add(bird_node)

def build_tree(viewer, shader):
    # Pathway trees
    tex_list = ["./../resources/FantasyWorld/Textures/Nature_Atlas_1.tga"]
    tree_size = 0.6
    # for i in range(-70, 100, 40):
    tree_node = Node(
        transform=translate(53, camera.cameraPositionXZ() + 3, 50.2) @ scale(tree_size, tree_size, tree_size) @ rotate((1, 0, 0), -90))
    mesh_list = multi_load_textured(file="./../resources/FantasyWorld/Constructable_Elements/Barrel_02.FBX", shader=shader,
                                    tex_file=tex_list,
                                    k_a=(.4, .4, .4),
                                    k_d=(1.2, 1.2, 1.2),
                                    k_s=(.2, .2, .2),
                                    s=4
                                    )
    for mesh in mesh_list:
        tree_node.add(mesh)
    viewer.add(tree_node)

    tree_node = Node(
    transform=translate(53, camera.cameraPositionXZ() + 5, 50.2) @ scale(tree_size, tree_size, tree_size) @ rotate((1, 0, 0), -90))
    mesh_list = multi_load_textured(file="./../resources/FantasyWorld/Constructable_Elements/Barrel_01.FBX", shader=shader,
                                    tex_file=tex_list,
                                    k_a=(.4, .4, .4),
                                    k_d=(1.2, 1.2, 1.2),
                                    k_s=(.2, .2, .2),
                                    s=4
                                    )
    for mesh in mesh_list:
        tree_node.add(mesh)
    viewer.add(tree_node)


    tree_size = 0.8
    # for i in range(-70, 100, 40):
    tree_node = Node(
        transform=translate(93, camera.cameraPositionXZ() + 3, 50.2) @ scale(tree_size, tree_size, tree_size) @ rotate((1, 0, 0), -90))
    mesh_list = multi_load_textured(file="./../resources/FantasyWorld/Constructable_Elements/HouseMushroom.FBX", shader=shader,
                                    tex_file=tex_list,
                                    k_a=(.4, .4, .4),
                                    k_d=(1.2, 1.2, 1.2),
                                    k_s=(.2, .2, .2),
                                    s=4
                                    )
    for mesh in mesh_list:
        tree_node.add(mesh)
    viewer.add(tree_node)

    tree_node = Node(
    transform=translate(96, camera.cameraPositionXZ() + 5, 50.2) @ scale(tree_size, tree_size, tree_size) @ rotate((1, 0, 0), -90))
    mesh_list = multi_load_textured(file="./../resources/FantasyWorld/Constructable_Elements/HouseMushroom_Window.FBX", shader=shader,
                                    tex_file=tex_list,
                                    k_a=(.4, .4, .4),
                                    k_d=(1.2, 1.2, 1.2),
                                    k_s=(.2, .2, .2),
                                    s=4
                                    )
    for mesh in mesh_list:
        tree_node.add(mesh)
    viewer.add(tree_node)



    # HOUSE SMALL IN SCENE  
    tree_size = 0.4
    # for i in range(-70, 100, 40):
    tree_node = Node(
        transform=translate(53, camera.cameraPositionXZ() + 3, 90.2) @ scale(tree_size, tree_size, tree_size) @ rotate((1, 0, 0), -90))
    mesh_list = multi_load_textured(file="./../resources/FantasyWorld/Constructable_Elements/HouseMushroom.FBX", shader=shader,
                                    tex_file=tex_list,
                                    k_a=(.4, .4, .4),
                                    k_d=(1.2, 1.2, 1.2),
                                    k_s=(.2, .2, .2),
                                    s=4
                                    )
    for mesh in mesh_list:
        tree_node.add(mesh)
    viewer.add(tree_node)

    tree_node = Node(
    transform=translate(55, camera.cameraPositionXZ() + 5, 90.2) @ scale(tree_size, tree_size, tree_size) @ rotate((1, 0, 0), -90))
    mesh_list = multi_load_textured(file="./../resources/FantasyWorld/Constructable_Elements/HouseMushroom_Window.FBX", shader=shader,
                                    tex_file=tex_list,
                                    k_a=(.4, .4, .4),
                                    k_d=(1.2, 1.2, 1.2),
                                    k_s=(.2, .2, .2),
                                    s=4
                                    )
    for mesh in mesh_list:
        tree_node.add(mesh)
    viewer.add(tree_node)

    # tree_node = Node(
    #     transform=translate(13, camera.cameraPositionXZ() + 3, 40) @ scale(tree_size, tree_size, tree_size) @ rotate((1, 0, 0), -90))
    # mesh_list = multi_load_textured(file="./../resources/FantasyWorld/Constructable_Elements/TreeStump.FBX", shader=shader,
    #                                 tex_file=tex_list,
    #                                 k_a=(.4, .4, .4),
    #                                 k_d=(1.2, 1.2, 1.2),
    #                                 k_s=(.2, .2, .2),
    #                                 s=4
    #                                 )
    # for mesh in mesh_list:
    #     tree_node.add(mesh)
    # viewer.add(tree_node)