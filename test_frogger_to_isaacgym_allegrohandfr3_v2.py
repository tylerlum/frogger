# %%
import numpy as np
from typing import Tuple
import trimesh
from pydrake.math import RigidTransform, RotationMatrix
from pydrake.all import Quaternion
from tqdm import tqdm
import pytorch_kinematics as pk
import torch

from frogger import ROOT
from frogger.objects import MeshObjectConfig, MeshObject
from frogger.robots.robots import AlgrModelConfig, FR3AlgrZed2iModelConfig
from frogger.sampling import HeuristicAlgrICSampler, HeuristicFR3AlgrICSampler
from frogger.solvers import FroggerConfig
from frogger.robots.robot_core import RobotModel
import plotly.graph_objects as go
import pathlib
from dataclasses import dataclass


# %%
@dataclass
class Args:
    obj_filepath: pathlib.Path = pathlib.Path(
        # "/juno/u/tylerlum/github_repos/nerf_grasping/nerf_grasping/baselines/nerf_meshdata/core-bottle-194f4eb1707aaf674c8b72e8da0e65c5/coacd/decomposed.obj"
        "/juno/u/tylerlum/github_repos/DexGraspNet/data/meshdata/core-bottle-2927d6c8438f6e24fe6460d8d9bd16c6/coacd/decomposed.obj"
    )
    obj_scale: float = 0.0915
    num_grasps: int = 3
    # obj_name: str = "core-bottle-194f4eb1707aaf674c8b72e8da0e65c5"
    obj_name: str = "core-bottle-2927d6c8438f6e24fe6460d8d9bd16c6"
    obj_is_yup: bool = True
    hand: str = "rh"
    robot_model_path: pathlib.Path = (
        pathlib.Path(ROOT) / "models/fr3_algr_zed2i/fr3_algr_zed2i.urdf"
    )

    @property
    def wrist_body_name(self) -> str:
        if self.hand == "lh":
            return "algr_lh_palm"
        elif self.hand == "rh":
            return "algr_rh_palm"
        else:
            raise ValueError(f"Invalid hand: {self.hand}")


args = Args()
# args = Args(
#     obj_filepath=pathlib.Path(
#         ROOT + "/data/001_chips_can/001_chips_can_clean.obj"
#     ),
#     obj_scale=1.0,
#     obj_name="001_chips_can",
#     obj_is_yup=False,
# )


# %%
def plot_mesh(mesh: trimesh.Trimesh) -> None:
    fig = go.Figure(
        data=[
            go.Mesh3d(
                x=mesh.vertices[:, 0],
                y=mesh.vertices[:, 1],
                z=mesh.vertices[:, 2],
                i=mesh.faces[:, 0],
                j=mesh.faces[:, 1],
                k=mesh.faces[:, 2],
                color="lightpink",
                opacity=0.50,
            )
        ]
    )
    fig.show()


# %%
def create_mesh(obj_filepath: pathlib.Path, obj_scale: float) -> trimesh.Trimesh:
    mesh = trimesh.load(obj_filepath)
    mesh.apply_transform(trimesh.transformations.scale_matrix(obj_scale))
    return mesh


# %%
mesh = create_mesh(obj_filepath=args.obj_filepath, obj_scale=args.obj_scale)

# %%
plot_mesh(mesh)


# %%
def compute_X_W_O(mesh: trimesh.Trimesh, obj_is_yup: bool) -> np.ndarray:
    bounds = mesh.bounds
    X_W_O = np.eye(4)
    if obj_is_yup:
        min_y_O = bounds[0, -2]
        X_W_O[:3, 3] = np.array([0.7, 0.0, -min_y_O])
        X_W_O[:3, :3] = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])[
            :3, :3
        ]
    else:
        min_z_O = bounds[0, -1]
        X_W_O[:3, 3] = np.array([0.7, 0.0, -min_z_O])
        X_W_O[:3, :3] = np.eye(3)
    return X_W_O


# %%
X_W_O = compute_X_W_O(mesh=mesh, obj_is_yup=args.obj_is_yup)


def create_mesh_object(
    mesh: trimesh.Trimesh, obj_name: str, X_W_O: np.ndarray
) -> MeshObject:
    return MeshObjectConfig(
        X_WO=RigidTransform(
            RotationMatrix(X_W_O[:3, :3]),
            X_W_O[:3, 3],
        ),
        mesh=mesh,
        name=obj_name,
        clean=False,
    ).create()


mesh_object = create_mesh_object(mesh=mesh, obj_name=args.obj_name, X_W_O=X_W_O)


# %%
def create_model(mesh_object: MeshObject, viz: bool = False) -> RobotModel:
    return FR3AlgrZed2iModelConfig(
        obj=mesh_object,
        ns=4,
        mu=0.7,
        d_min=0.001,
        d_pen=0.005,
        viz=viz,
    ).create()


# %%
def zup_mesh_to_q_array(mesh_object: MeshObject, num_grasps: int) -> np.ndarray:
    # loading model
    model = create_model(mesh_object=mesh_object, viz=False)

    # creating solver and generating grasp
    frogger = FroggerConfig(
        model=model,
        sampler=HeuristicFR3AlgrICSampler(model, z_axis_fwd=True),
        tol_surf=1e-3,
        tol_joint=1e-2,
        tol_col=1e-3,
        tol_fclosure=1e-5,
        xtol_rel=1e-6,
        xtol_abs=1e-6,
        maxeval=1000,
        maxtime=60.0,
    ).create()

    print("Model compiled! Generating grasp...")
    q_array = []
    for _ in tqdm(range(num_grasps)):
        q_star = frogger.generate_grasp()
        assert q_star is not None
        assert q_star.shape == (23,)
        q_array.append(q_star)
    q_array = np.array(q_array)
    assert q_array.shape == (num_grasps, 23)
    return q_array


# %%
q_array = zup_mesh_to_q_array(mesh_object=mesh_object, num_grasps=args.num_grasps)


# %%
def visualize_q_with_pydrake(mesh_object: MeshObject, q: np.ndarray) -> None:
    assert q.shape == (23,)
    # loading model
    model = create_model(mesh_object=mesh_object, viz=True)
    model.viz_config(q)


# %%
IDX = 1
# visualize_q_with_pydrake(mesh_object=mesh_object, q=q_array[IDX])


# %%
def add_transform_traces(
    fig: go.Figure, T: np.ndarray, name: str, length: float = 0.02
) -> None:
    assert T.shape == (4, 4)
    origin = np.array([0.0, 0.0, 0.0, 1.0])
    x_axis = np.array([length, 0.0, 0.0, 1.0])
    y_axis = np.array([0.0, length, 0.0, 1.0])
    z_axis = np.array([0.0, 0.0, length, 1.0])

    origin = T @ origin
    x_axis = T @ x_axis
    y_axis = T @ y_axis
    z_axis = T @ z_axis

    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], x_axis[0]],
            y=[origin[1], x_axis[1]],
            z=[origin[2], x_axis[2]],
            mode="lines",
            line=dict(color="red"),
            name=f"{name}_x",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], y_axis[0]],
            y=[origin[1], y_axis[1]],
            z=[origin[2], y_axis[2]],
            mode="lines",
            line=dict(color="green"),
            name=f"{name}_y",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[origin[0], z_axis[0]],
            y=[origin[1], z_axis[1]],
            z=[origin[2], z_axis[2]],
            mode="lines",
            line=dict(color="blue"),
            name=f"{name}_z",
        )
    )


# %%
def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    B = points.shape[0]
    assert points.shape == (B, 3)
    assert T.shape == (4, 4)

    points = np.hstack((points, np.ones((B, 1))))
    assert points.shape == (B, 4)
    points = T @ points.T
    points = points.T[:, :3]
    return points


vertices_O = mesh.vertices
vertices_W = transform_points(points=vertices_O, T=X_W_O)
fig = go.Figure()
fig.update_layout(scene=dict(aspectmode="data"), title=dict(text="W frame"))
fig.add_trace(
    go.Mesh3d(
        x=vertices_W[:, 0],
        y=vertices_W[:, 1],
        z=vertices_W[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color="lightpink",
        opacity=0.50,
    )
)
add_transform_traces(fig=fig, T=np.eye(4), name="T_W")
add_transform_traces(fig=fig, T=X_W_O, name="T_O")
fig.show()


# %%
def get_kinematic_chain(model_path: pathlib.Path) -> pk.Chain:
    with open(model_path) as f:
        chain = pk.build_chain_from_urdf(f.read())
        chain = chain.to(device="cuda", dtype=torch.float32)
    return chain


chain = get_kinematic_chain(model_path=args.robot_model_path)


# %%
def q_to_T_W_H_and_joint_angles(
    q: np.ndarray, chain: pk.Chain, wrist_body_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    assert q.shape == (23,)
    hand_joint_angles = q[7:23]

    link_poses_hand_frame = chain.forward_kinematics(q)
    X_W_Wrist = (
        link_poses_hand_frame[wrist_body_name].get_matrix().squeeze(dim=0).cpu().numpy()
    )

    assert X_W_Wrist.shape == (4, 4)
    assert hand_joint_angles.shape == (16,)
    return X_W_Wrist, hand_joint_angles


# %%
X_W_Wrist, _ = q_to_T_W_H_and_joint_angles(
    q=q_array[IDX], chain=chain, wrist_body_name=args.wrist_body_name
)
add_transform_traces(fig=fig, T=X_W_Wrist, name="T_Wrist")
fig.show()

# %%
if args.obj_is_yup:
    X_O_Oy = np.eye(4)
else:
    X_O_Oy = trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0])

model = create_model(mesh_object=mesh_object, viz=False)
X_O_Wrist = np.linalg.inv(X_W_O) @ X_W_Wrist
X_Oy_Wrist = np.linalg.inv(X_O_Oy) @ X_O_Wrist

# %%
link_poses_hand_frame = chain.forward_kinematics(q_array[IDX])
FINGERTIP_LINK_NAMES = [
    "algr_rh_if_ds_tip",
    "algr_rh_mf_ds_tip",
    "algr_rh_rf_ds_tip",
    "algr_rh_th_ds_tip",
]
X_W_fingertip_list = [
    link_poses_hand_frame[ln].get_matrix().squeeze(dim=0).cpu().numpy()
    for ln in FINGERTIP_LINK_NAMES
]
for i, X_W_fingertip in enumerate(X_W_fingertip_list):
    add_transform_traces(fig=fig, T=X_W_fingertip, name=f"T_fingertip {i}")
fig.show()


# %%
# %%
def q_array_to_hand_config_dict(
    q_array: np.ndarray, X_W_O: np.ndarray, X_O_Oy: np.ndarray
) -> dict:
    # W = world frame z-up
    # O = object frame z-up
    # Oy = object frame y-up
    # H = hand/frame z along finger, x away from palm
    # Assumes q in W frame
    # Assumes hand_config_dict in Oy frame

    B = q_array.shape[0]
    assert q_array.shape == (B, 23)
    assert X_W_O.shape == (4, 4)
    assert X_O_Oy.shape == (4, 4)

    X_W_H_array, joint_angles_array = [], []
    for i in range(B):
        X_W_H, joint_angles = q_to_T_W_H_and_joint_angles(
            q=q_array[i], chain=chain, wrist_body_name=args.wrist_body_name
        )
        assert X_W_H.shape == (4, 4)
        assert joint_angles.shape == (16,)
        X_W_H_array.append(X_W_H)
        joint_angles_array.append(joint_angles)

    X_W_H_array = np.array(X_W_H_array)
    joint_angles_array = np.array(joint_angles_array)
    assert X_W_H_array.shape == (B, 4, 4)
    assert joint_angles_array.shape == (B, 16)

    X_O_W = np.linalg.inv(X_W_O)
    X_Oy_O = np.linalg.inv(X_O_Oy)

    X_Oy_H_array = []
    for i in range(B):
        X_Oy_H = X_Oy_O @ X_O_W @ X_W_H_array[i]
        X_Oy_H_array.append(X_Oy_H)
    X_Oy_H_array = np.array(X_Oy_H_array)

    return {
        "trans": X_Oy_H_array[:, :3, 3],
        "rot": X_Oy_H_array[:, :3, :3],
        "joint_angles": joint_angles_array,
    }


# %%
# def add_line_trace(
#     fig: go.Figure,
#     start_point: np.ndarray,
#     direction: np.ndarray,
#     name: str,
#     length: float = 0.02,
# ) -> None:
#     assert start_point.shape == (3,)
#     assert direction.shape == (3,)
#     assert np.allclose(np.linalg.norm(direction), 1.0)
# 
#     end_point = start_point + length * direction
#     print(f"start_point: {start_point}")
#     print(f"end_point: {end_point}")
#     print(f"direction: {direction}")
#     print()
# 
#     fig.add_trace(
#         go.Scatter3d(
#             x=[start_point[0], end_point[0]],
#             y=[start_point[1], end_point[1]],
#             z=[start_point[2], end_point[2]],
#             mode="lines",
#             line=dict(color="black"),
#             name=name,
#         )
#     )
# 
# 
# assert model.n_O.shape == (3, 4)
# assert model.n_W.shape == (3, 4)
# for i in range(model.n_O.shape[1]):
#     assert np.allclose(np.linalg.norm(model.n_O[:, i]), 1.0)
#     assert np.allclose(np.linalg.norm(model.n_W[:, i]), 1.0)
# 
# # for i in range(model.n_O.shape[1]):
# #     start_position = X_W_fingertip_list[i][:3, 3]
# #     direction = model.n_O[:, i]
# #     add_line_trace(fig=fig, start_point=start_position, direction=direction, name=f"n_W {i}")
# for i in range(model.n_W.shape[1]):
#     start_position = X_W_fingertip_list[i][:3, 3]
#     direction = model.n_W[:, i]
#     print(f"n_W {i}: {direction}")
#     add_line_trace(
#         fig=fig, start_point=start_position, direction=direction, name=f"n_W {i}"
#     )
# 
# fig.show()

# %%

model.n_W
# %%
model.n_O
# %%

# %%
q_array.shape
# %%
model.n_O, model.R_cf_O


# %%
def R_p_to_T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    print(f"R: {R}")
    assert R.shape == (3, 3)
    assert p.shape == (3,)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


# %%
assert model.R_cf_O.shape == (4, 3, 3)
for i in range(model.R_cf_O.shape[0]):
    # add_transform_traces(fig=fig, T=R_p_to_T(R=model.R_cf_O[i], p=X_W_fingertip_list[i][:3, 3]), name=f"R_cf_O {i}")
    add_transform_traces(fig=fig, T=X_W_fingertip_list[i] @ R_p_to_T(R=model.R_cf_O[i], p=np.zeros(3)), name=f"R_cf_O {i}")
    # add_transform_traces(fig=fig, T=X_W_O @ R_p_to_T(R=model.R_cf_O[i], p=np.zeros(3)), name=f"R_cf_O {i}")
    # add_transform_traces(fig=fig, T=R_p_to_T(R=X_W_O[:3, :3] @ model.R_cf_O[i], p=X_W_fingertip_list[i][:3, 3]), name=f"R_cf_O {i}")
fig.show()

# %%
model.n_O, model.R_cf_O

# %%

# %%
visualize_q_with_pydrake(mesh_object=mesh_object, q=q_array[IDX])

# %%
X_W_O[:3, :3]
# %%
print(f"model.R_cf_O = \n{model.R_cf_O}")

# %%
hand_config_dict = q_array_to_hand_config_dict(
    q_array=q_array, X_W_O=X_W_O, X_O_Oy=X_O_Oy
)

# %%
output_folder = pathlib.Path(".") / "output_hand_config_dicts"
output_folder.mkdir(exist_ok=True)
np.save(
    # Convert 0.0915 to 0_0915 (always 4 decimal places)
    output_folder / f"{args.obj_name}_{args.obj_scale:.4f}".replace(".", "_"),
    hand_config_dict,
    allow_pickle=True,
)

# %%
hand_config_dict['joint_angles'].shape

# %%
hand_config_dict['trans']

# %%
visualize_q_with_pydrake(mesh_object=mesh_object, q=q_array[2])

# %%
X_Oy_O = np.linalg.inv(X_O_Oy)
X_O_W = np.linalg.inv(X_W_O)

fig = go.Figure()
# Set title to be Oy frame
fig.update_layout(scene=dict(aspectmode="data"), title=dict(text="Oy frame"))
vertices_Oy = transform_points(points=vertices_O, T=X_Oy_O)
fig = go.Figure()
fig.add_trace(
    go.Mesh3d(
        x=vertices_Oy[:, 0],
        y=vertices_Oy[:, 1],
        z=vertices_Oy[:, 2],
        i=mesh.faces[:, 0],
        j=mesh.faces[:, 1],
        k=mesh.faces[:, 2],
        color="lightpink",
        opacity=0.50,
    )
)
add_transform_traces(fig=fig, T=np.eye(4), name="T_Oy")
fig.show()
# %%
add_transform_traces(fig=fig, T=X_Oy_O @ X_O_W @ X_W_Wrist, name="T_Wrist")
for i, X_W_fingertip in enumerate(X_W_fingertip_list):
    add_transform_traces(fig=fig, T=X_Oy_O @ X_O_W @ X_W_fingertip, name=f"T_fingertip {i}")
fig.show()

# %%
