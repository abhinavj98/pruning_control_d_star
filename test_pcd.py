from gym_env_discrete import ur5GymEnv
import pybullet
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt 

def get_point_cloud(env):
    """Deprojects the depth image to a point cloud in world coordinates."""
    width, height, view_matrix, proj_matrix = 1024, 1024, env.view_mat, env.proj_mat
    depth = env.depth
    proj_matrix = np.asarray(proj_matrix).reshape([4, 4], order="F")
    view_matrix = np.asarray(view_matrix).reshape([4, 4], order="F")
    tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
    y, x = np.mgrid[-1:1:2 / height, -1:1:2 / width]
    y *= -1.
    x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
    h = np.ones_like(z)
    pixels = np.stack([x,y, z, h], axis=1)
    # filter out "infinite" depths
    pixels = pixels[z < 0.99]
    pixels[:, 2] = 2 * pixels[:, 2] - 1
    # turn pixels to world coordinates
    points = np.matmul(tran_pix_world, pixels.T).T
    points /= points[:, 3: 4]
    points = points[:, :3]
    return points

def pcd_to_occupancy(pcd: np.ndarray):
    """Discretizes the point cloud into a 3D occupancy grid."""
    if pcd.shape[1] != 3:
        raise ValueError(f'pcd should have 3 columns, instead got {pcd.shape[1]}')
    
    pcd = (pcd*100).round()

    x_min = pcd[:, 0].min()
    x_max = pcd[:, 0].max()

    y_min = pcd[:, 1].min()
    y_max = pcd[:, 1].max()

    z_min = pcd[:, 2].min()
    z_max = pcd[:, 2].max()


    occupancy = np.zeros((int(x_max - x_min + 1), int(y_max-y_min+1), int(z_max - z_min + 1)), dtype=np.float64)

    for i in range(pcd.shape[0]):
        x = int(pcd[i, 0] - x_min)
        y = int(pcd[i, 1] - y_min)
        z = int(pcd[i, 2] - z_min)

        occupancy[x, y,z] += 1

    # occupancy /= pcd.shape[0]
    return occupancy >= 1

def increment_map_update(map, pcd):
    """Adds point clouds to create a gloabl map."""
    if map is None:
        map = np.empty((1,3)).astype(np.float64)
    map = np.append(pcd, map, axis=0)
    return map

if __name__ == "__main__":
    tree_urdf_path = "./ur_e_description/urdf/trees/train"
    tree_obj_path = "./ur_e_description/meshes/trees/train"
    #Render arm movements in pybullet (For visualization)
    renders = True
    env = ur5GymEnv(tree_urdf_path = tree_urdf_path, tree_obj_path=tree_obj_path, renders=renders)
    env.reset()

    map = None
    step = 0
    random = True
    
    while True:
        # Take 10 random actions to get a view of the tree
        if random:
            action = np.random.randint(0, 12)
            if step > 10:
                action = 13
            step+=1
            print(step)
        # else:
        #     keys = pybullet.getKeyboardEvents()
        #     action = 0
        #     for k,state in keys.items():
        #         if ord('-') == k:
        #             action = 11
        #         elif ord('=') == k:
        #             action = 12
        #         else:
        #             action = (k - ord('0'))
        #             # if action == 0:
        #             #     action = 10
        if action > 12 or action < 0:
            print("Bad action")
            break
        # if not random:
        #     if state&pybullet.KEY_WAS_TRIGGERED:
        #         print(env.rev_actions[action])
        #         r = env.step(action, False)
        r = env.step(action, False)
        map = increment_map_update(map, get_point_cloud(env))
    

    # Visualize map as a point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(map)
    pcd.transform([[ 1,0,0,0], [0, 0,1,0], [0, 1,0, 0], [0, 0, 0, 1]])
    viewer = o3d.visualization.Visualizer()
    viewer.create_window()
    viewer.add_geometry(pcd)
    opt = viewer.get_render_option()
    opt.show_coordinate_frame = True
    opt.background_color = np.asarray([0.5, 0.5, 0.5])
    viewer.run()
    viewer.destroy_window()
    
    # Visualize map as a voxel grid
    occ = pcd_to_occupancy(map)
    print(occ.shape)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.voxels(occ, edgecolor="k", facecolors='green')
    plt.show()