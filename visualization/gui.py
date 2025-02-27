import pathlib
import threading
import time
from datetime import datetime
import cv2
import glfw
import imgviz
import numpy as np
import open3d as o3d
import os
import pickle
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import torch
import torch.nn.functional as F
from OpenGL import GL as gl

from utils.operations import GaussianRenderer
from utils.common import Camera, Gui2Mapper, Mapper2Gui
from .gl_render import util, util_gau
from .gl_render.render_ogl import OpenGLRenderer
from .gui_utils import (
    create_frustum,
    create_path,
    create_voxel,
    get_latest_queue,
    fov2focal,
    cv_gl,
    gl_cv,
    c2w_to_lookat,
    model_matrix_to_extrinsic_matrix,
    create_camera_intrinsic_from_size,
)

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


class GUI:
    def __init__(self, cfg=None, params_gui=None):
        self.step = 0
        self.process_finished = False
        self.device = "cuda"

        # default setup
        if cfg is None:
            self.window_h, self.window_w = [1200, 2100]
            self.render_h, self.render_w = [512, 512]
            self.near, self.far = [0.1, 10]
            self.fov = [60, 60]
            self.background = [0.0, 0.0, 0.0, 1.0]
        else:
            self.window_h, self.window_w = cfg.resolution_win
            self.render_h, self.render_w = cfg.resolution_render
            self.near, self.far = cfg.bound
            self.fov = cfg.fov
            self.background = cfg.background

        self.init_widget()

        self.init = False
        self.require_rasterization = True
        self.show_view_candidates = False
        self.record_on = False
        self.loaded_camera_path = None
        self.kf_window = None
        self.render_img = None
        self.q_mapper2gui = None
        self.q_gui2mapper = None
        self.q_planner2gui = None
        self.q_gui2planner = None

        self.gaussian_cur = None
        self.voxel_cur = None
        self.mesh_cur = None
        self.cam_path = []
        self.gaussian_nums = []
        self.frame_dict = {}
        self.voxel_type = "none"

        if params_gui is not None:
            self.q_mapper2gui = (
                params_gui["mapper_receive"]
                if "mapper_receive" in params_gui.keys()
                else None
            )
            self.q_gui2mapper = (
                params_gui["mapper_send"]
                if "mapper_send" in params_gui.keys()
                else None
            )
            self.q_planner2gui = (
                params_gui["planner_receive"]
                if "planner_receive" in params_gui.keys()
                else None
            )
            self.q_gui2planner = (
                params_gui["planner_send"]
                if "planner_send" in params_gui.keys()
                else None
            )

        self.g_camera = util.Camera(self.window_h, self.window_w)
        self.window_gl = self.init_glfw()
        self.g_renderer = OpenGLRenderer(self.g_camera.w, self.g_camera.h)

        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)
        self.gaussians_gl = util_gau.GaussianData(0, 0, 0, 0, 0)

        save_path = "outputs_gui"
        save_path = pathlib.Path(save_path)

        self.pose_save_path = save_path / "saved_poses"
        self.path_save_path = save_path / "saved_paths"
        self.screenshot_save_path = save_path / "screenshots"
        os.makedirs(self.pose_save_path, exist_ok=True)
        os.makedirs(self.path_save_path, exist_ok=True)
        os.makedirs(self.screenshot_save_path, exist_ok=True)

        threading.Thread(target=self._update_thread).start()
        self.gui_loaded = True

    def init_widget(self):
        self.window = gui.Application.instance.create_window(
            "ActiveGS", self.window_w, self.window_h
        )
        self.window.set_on_layout(self._on_layout)
        self.window.set_on_close(self._on_close)
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(self.window.renderer)
        self.widget3d.scene.set_background(self.background, None)

        cg_settings = rendering.ColorGrading(
            rendering.ColorGrading.Quality.ULTRA,
            rendering.ColorGrading.ToneMapping.LINEAR,
        )
        self.widget3d.scene.view.set_color_grading(cg_settings)
        self.window.add_child(self.widget3d)

        self.mesh_render = rendering.MaterialRecord()
        self.mesh_render.shader = "normals"
        # self.mesh_render.shader = "defaults"

        self.lit_line = rendering.MaterialRecord()
        self.lit_line.shader = "unlitLine"
        self.lit_line.line_width = 4.0

        self.lit_dot = rendering.MaterialRecord()
        self.lit_dot.point_size = 6.0

        self.lit_vox = rendering.MaterialRecord()
        self.lit_vox.shader = "unlitLine"
        self.lit_vox.line_width = 1.0

        self.lit = rendering.MaterialRecord()
        self.lit.shader = "defaultUnlit"

        # bounds = self.widget3d.scene.bounding_box
        # self.widget3d.setup_camera(60.0, bounds, bounds.get_center())
        em = self.window.theme.font_size
        margin = 0.5 * em
        self.panel = gui.Vert(0.5 * em, gui.Margins(margin))

        self.button_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.pause_button = gui.ToggleSwitch("Resume/Pause")
        self.pause_button.is_on = False
        self.pause_button.set_on_clicked(self._on_pause_button)
        self.button_tile.add_child(self.pause_button)

        self.record_button = gui.ToggleSwitch("Record/Stop")
        self.record_button.is_on = False
        self.record_button.set_on_clicked(self._on_record_button)
        self.button_tile.add_child(self.record_button)
        self.panel.add_child(self.button_tile)

        self.panel.add_child(gui.Label("Viewpoint Options"))

        viewpoint_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        vp_subtile1 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile2 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile3 = gui.Vert(0.5 * em, gui.Margins(margin))
        vp_subtile4 = gui.Vert(0.5 * em, gui.Margins(margin))

        ##Check boxes
        vp_subtile1.add_child(gui.Label("Camera Follow Options"))
        chbox_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.followcam_chbox = gui.Checkbox("Follow Camera")
        self.followcam_chbox.checked = False
        chbox_tile.add_child(self.followcam_chbox)

        self.staybehind_chbox = gui.Checkbox("From Behind")
        self.staybehind_chbox.checked = False
        chbox_tile.add_child(self.staybehind_chbox)

        # NOTE: in fly mode, you can control like a game using WASD,Q,Z,E,R, up, right, left, down
        self.flycam_chbox = gui.Checkbox("Fly")
        self.flycam_chbox.checked = False
        self.flycam_chbox.set_on_checked(self._set_control_mode)
        chbox_tile.add_child(self.flycam_chbox)
        vp_subtile1.add_child(chbox_tile)

        ##Combo panels
        ## Jump to the training camera viewpoint
        combo_tile_kf = gui.Vert(0.5 * em, gui.Margins(margin))
        self.combo_kf = gui.Combobox()
        self.combo_kf.set_on_selection_changed(self._on_combo_kf)
        combo_tile_kf.add_child(gui.Label("History Views"))
        combo_tile_kf.add_child(self.combo_kf)
        vp_subtile2.add_child(combo_tile_kf)

        # gui camera pose setup
        vp_subtile3.add_child(gui.Label("Camera Pose"))
        combo_tile_camera_pose = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.combo_camera_pose = gui.Combobox()

        # maximum 10 camera poses can be saved
        for i in range(10):
            self.combo_camera_pose.add_item(str(i))
        combo_tile_camera_pose.add_child(self.combo_camera_pose)

        self.save_pose_btn = gui.Button("Save")
        self.save_pose_btn.set_on_clicked(self._on_save_pose_btn)
        combo_tile_camera_pose.add_child(self.save_pose_btn)

        self.load_pose_btn = gui.Button("Load")
        self.load_pose_btn.set_on_clicked(self._on_load_pose_btn)
        combo_tile_camera_pose.add_child(self.load_pose_btn)
        vp_subtile3.add_child(combo_tile_camera_pose)

        # gui camera path setup
        vp_subtile4.add_child(gui.Label("Camera Path"))
        combo_tile_camera_path = gui.Horiz(0.5 * em, gui.Margins(margin))
        self.combo_camera_path = gui.Combobox()

        # maximum 5 camera paths can be saved
        for i in range(5):
            self.combo_camera_path.add_item(str(i))
        combo_tile_camera_path.add_child(self.combo_camera_path)

        self.reset_path_btn = gui.Button("Reset")
        self.reset_path_btn.set_on_clicked(self._on_reset_path_btn)
        combo_tile_camera_path.add_child(self.reset_path_btn)

        vp_subtile4.add_child(combo_tile_camera_path)

        viewpoint_tile.add_child(vp_subtile1)
        viewpoint_tile.add_child(vp_subtile2)
        viewpoint_tile.add_child(vp_subtile3)
        viewpoint_tile.add_child(vp_subtile4)

        self.panel.add_child(viewpoint_tile)

        self.panel.add_child(gui.Label("3D Objects"))
        chbox_tile_3dobj = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.axis_chbox = gui.Checkbox("Axis")
        self.axis_chbox.checked = False
        self.axis_chbox.set_on_checked(self._on_axis_chbox)
        chbox_tile_3dobj.add_child(self.axis_chbox)

        self.cameras_chbox = gui.Checkbox("Cameras")
        self.cameras_chbox.checked = True
        self.cameras_chbox.set_on_checked(self._on_cameras_chbox)
        chbox_tile_3dobj.add_child(self.cameras_chbox)

        self.path_chbox = gui.Checkbox("Path")
        self.path_chbox.checked = True
        self.path_chbox.set_on_checked(self._on_path_chbox)
        chbox_tile_3dobj.add_child(self.path_chbox)

        self.vc_chbox = gui.Checkbox("View Candidates")
        self.vc_chbox.checked = False
        self.vc_chbox.set_on_checked(self._on_vc_chbox)
        chbox_tile_3dobj.add_child(self.vc_chbox)

        self.pcd_chbox = gui.Checkbox("Point Cloud")
        self.pcd_chbox.checked = False
        self.pcd_chbox.set_on_checked(self._on_pcd_chbox)
        chbox_tile_3dobj.add_child(self.pcd_chbox)

        self.mesh_chbox = gui.Checkbox("Mesh")
        self.mesh_chbox.checked = False
        self.mesh_chbox.set_on_checked(self._on_mesh_chbox)
        chbox_tile_3dobj.add_child(self.mesh_chbox)

        combo_voxel_tile = gui.Vert(0.5 * em, gui.Margins(margin))
        self.combo_voxel = gui.Combobox()
        self.combo_voxel.set_on_selection_changed(self._on_combo_voxel)
        for voxel_type in [
            "none",
            "occ",
            "free",
            "unknown",
            "planning",
            "unexplored",
            "frontier",
            "roi",
        ]:
            self.combo_voxel.add_item(voxel_type)
        combo_voxel_tile.add_child(gui.Label("Voxel Map"))
        combo_voxel_tile.add_child(self.combo_voxel)
        chbox_tile_3dobj.add_child(combo_voxel_tile)

        self.panel.add_child(chbox_tile_3dobj)

        self.panel.add_child(gui.Label("Rendering options"))
        chbox_tile_geometry = gui.Horiz(0.5 * em, gui.Margins(margin))

        self.depth_chbox = gui.Checkbox("Depth")
        self.depth_chbox.checked = False
        chbox_tile_geometry.add_child(self.depth_chbox)

        self.confidence_chbox = gui.Checkbox("Confidence")
        self.confidence_chbox.checked = False
        chbox_tile_geometry.add_child(self.confidence_chbox)

        self.opacity_chbox = gui.Checkbox("Opacity")
        self.opacity_chbox.checked = False
        chbox_tile_geometry.add_child(self.opacity_chbox)

        self.normal_chbox = gui.Checkbox("Normal")
        self.normal_chbox.checked = False
        chbox_tile_geometry.add_child(self.normal_chbox)

        self.d2n_chbox = gui.Checkbox("D2N")
        self.d2n_chbox.checked = False
        chbox_tile_geometry.add_child(self.d2n_chbox)

        self.frontonly_chbox = gui.Checkbox("Surface Front")
        self.frontonly_chbox.checked = False
        chbox_tile_geometry.add_child(self.frontonly_chbox)

        self.bgcolor_chbox = gui.Checkbox("White Background")
        self.bgcolor_chbox.checked = False
        self.bgcolor_chbox.set_on_checked(self._on_bgcolor_chbox)
        chbox_tile_geometry.add_child(self.bgcolor_chbox)

        self.panel.add_child(chbox_tile_geometry)

        scale_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        scale_slider_label = gui.Label("Scale (0.1-1.0)")
        self.scale_slider = gui.Slider(gui.Slider.DOUBLE)
        self.scale_slider.set_limits(0.1, 1.0)
        self.scale_slider.double_value = 1.0
        scale_slider_tile.add_child(scale_slider_label)
        scale_slider_tile.add_child(self.scale_slider)
        self.panel.add_child(scale_slider_tile)

        ratio_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        ratio_slider_label = gui.Label("Rendering Ratio (0.1-1.0)")
        self.ratio_slider = gui.Slider(gui.Slider.DOUBLE)
        self.ratio_slider.set_limits(0.1, 1.0)
        self.ratio_slider.double_value = 1.0
        ratio_slider_tile.add_child(ratio_slider_label)
        ratio_slider_tile.add_child(self.ratio_slider)
        self.panel.add_child(ratio_slider_tile)

        thres_slider_tile = gui.Horiz(0.5 * em, gui.Margins(margin))
        thres_slider_label = gui.Label("Confidence Threshold (0.0-1.0)")
        self.thres_slider = gui.Slider(gui.Slider.DOUBLE)
        self.thres_slider.set_limits(0.0, 1.0)
        self.thres_slider.double_value = 1.0
        thres_slider_tile.add_child(thres_slider_label)
        thres_slider_tile.add_child(self.thres_slider)
        self.panel.add_child(thres_slider_tile)

        # screenshot button
        self.screenshot_btn = gui.Button("Screenshot")
        self.screenshot_btn.set_on_clicked(
            self._on_screenshot_btn
        )  # set the callback function
        self.panel.add_child(self.screenshot_btn)

        ## Rendering Tab
        tab_margins = gui.Margins(0, int(np.round(0.5 * em)), 0, 0)
        tabs = gui.TabControl()

        tab_info = gui.Vert(0, tab_margins)
        self.output_info = gui.Label("Number of Gaussians: ")
        tab_info.add_child(self.output_info)

        self.in_rgb_widget = gui.ImageWidget()
        self.in_depth_widget = gui.ImageWidget()
        tab_info.add_child(gui.Label("Input Color/Depth"))
        tab_info.add_child(self.in_rgb_widget)
        tab_info.add_child(self.in_depth_widget)

        tabs.add_tab("Info", tab_info)
        self.panel.add_child(tabs)
        self.window.add_child(self.panel)

        # add coordinate system
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=0.3, origin=[0, 0, 0]
        )
        self.widget3d.scene.add_geometry("axis", axis, self.lit)
        self.widget3d.scene.show_geometry("axis", self.axis_chbox.checked)

        # add invisible minimum bounding box
        bbox = o3d.geometry.AxisAlignedBoundingBox([-1.5, -1.5, -1.5], [1.5, 1.5, 1.5])
        self.widget3d.scene.add_geometry("bbox", bbox, self.lit)
        self.widget3d.scene.show_geometry("bbox", False)

    def init_glfw(self):
        window_name = "headless rendering"

        if not glfw.init():
            exit(1)

        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        window = glfw.create_window(
            self.window_w, self.window_h, window_name, None, None
        )
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        if not window:
            glfw.terminate()
            exit(1)
        return window

    # callback functions
    def _on_layout(self, layout_context):
        contentRect = self.window.content_rect
        self.widget3d_width_ratio = 1.0  # ratio to the window height
        self.widget3d_width = int(
            self.window.size.height * self.widget3d_width_ratio
        )  # 15 ems wide
        self.widget3d.frame = gui.Rect(
            contentRect.x, contentRect.y, self.widget3d_width, contentRect.height
        )  # square window
        self.panel.frame = gui.Rect(
            self.widget3d.frame.get_right(),
            contentRect.y,
            contentRect.width - self.widget3d_width,
            contentRect.height,
        )
        self.init_camera()

    def _on_close(self):
        self.is_done = True
        self.process_finished = True
        return True  # False would cancel the close

    def _set_control_mode(self, is_on):
        self.init_camera()
        if is_on:
            self.widget3d.set_view_controls(gui.SceneWidget.Controls.FLY)
        else:
            self.widget3d.set_view_controls(
                gui.SceneWidget.Controls.ROTATE_CAMERA_SPHERE
            )

    def _on_bgcolor_chbox(self, is_checked, name=None):
        if is_checked:
            self.background = [1.0, 1.0, 1.0, 0.0]
            self.widget3d.scene.set_background(self.background, None)
        else:
            self.background = [0.0, 0.0, 0.0, 0.0]
            self.widget3d.scene.set_background(self.background, None)

    def _on_combo_voxel(self, new_val, new_idx):
        self.voxel_type = new_val
        if self.voxel_cur is not None:
            self.update_voxel()
            self.widget3d.scene.show_geometry("voxel", True)
        else:
            print("voxel map not available!!!")

    def _on_reset_path_btn(self):
        saved_path_name = "saved_path_{}.txt".format(
            self.combo_camera_path.selected_text
        )
        saved_path_file = os.path.join(self.path_save_path, saved_path_name)
        print(saved_path_file)
        if os.path.exists(saved_path_file):
            os.remove(saved_path_file)

    def _on_load_pose_btn(self):
        saved_view_name = "saved_view_{}.pkl".format(
            self.combo_camera_pose.selected_text
        )
        saved_view_file = os.path.join(self.pose_save_path, saved_view_name)

        try:
            with open(saved_view_file, "rb") as pickle_file:
                saved_view = pickle.load(pickle_file)

            # setup_camera require w2c in opencv format
            self.widget3d.setup_camera(
                saved_view["intrinsic"],
                saved_view["extrinsic"],
                saved_view["width"],
                saved_view["height"],
                self.widget3d.scene.bounding_box,
            )
        except Exception as e:
            print("pose file not exist!!!")

    def _on_save_pose_btn(self):
        saved_view_name = "saved_view_{}.pkl".format(
            self.combo_camera_pose.selected_text
        )
        saved_view_file = os.path.join(self.pose_save_path, saved_view_name)
        try:
            extrinsic, intrinsic, resolution = self.gui_camera_params
            saved_view = dict(
                extrinsic=extrinsic,
                intrinsic=intrinsic,
                height=resolution[0],
                width=resolution[1],
            )
            with open(saved_view_file, "wb") as pickle_file:
                pickle.dump(saved_view, pickle_file)
        except Exception as e:
            print(e)

    def _on_combo_kf(self, new_val, new_idx):
        frustum = self.frame_dict[new_val]["frustum"]
        rgb = self.frame_dict[new_val]["rgb"]
        depth = self.frame_dict[new_val]["depth"]
        viewpoint = frustum.view_dir

        self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])
        if rgb is not None:
            self.in_rgb_widget.update_image(rgb)
        if depth is not None:
            self.in_depth_widget.update_image(depth)

    def _on_cameras_chbox(self, is_checked, name=None):
        for frame_name in self.frame_dict.keys():
            self.widget3d.scene.show_geometry(f"{frame_name}_camera", is_checked)

    def _on_path_chbox(self, is_checked, name=None):
        self.widget3d.scene.show_geometry("camera_path", is_checked)

    def _on_pcd_chbox(self, is_checked, name=None):
        name = "pcd"
        if is_checked:
            self.require_rasterization = False
            self.update_pcd()
            self.widget3d.scene.set_background([0, 0, 0, 1], None)
            self.widget3d.scene.show_geometry(name, True)
        else:
            self.widget3d.scene.show_geometry(name, False)
            self.require_rasterization = True

    def _on_mesh_chbox(self, is_checked, name=None):
        name = "mesh"
        if self.mesh_cur is not None:
            if is_checked:
                self.widget3d.scene.add_geometry(
                    "mesh", self.mesh_cur, self.mesh_render
                )
                self.require_rasterization = False
                self.widget3d.scene.set_background([0, 0, 0, 1], None)
                self.widget3d.scene.show_geometry(name, True)
            else:
                self.widget3d.scene.show_geometry(name, False)
                self.require_rasterization = True

    def _on_vc_chbox(self, is_checked, name=None):
        name_1 = "view_candidates_xyz"
        name_2 = "view_candidates_direction"
        if is_checked:
            self.show_view_candidates = True
            # self.update_view_candidates()
            self.widget3d.scene.show_geometry(name_1, True)
            self.widget3d.scene.show_geometry(name_2, True)
        else:
            self.show_view_candidates = False
            self.widget3d.scene.show_geometry(name_1, False)
            self.widget3d.scene.show_geometry(name_2, False)

    def _on_axis_chbox(self, is_checked):
        name = "axis"
        self.widget3d.scene.show_geometry(name, is_checked)

    def _on_pause_button(self, is_on):
        packet = Gui2Mapper()
        packet.flag_pause = self.pause_button.is_on
        self.q_gui2mapper.put(packet)

    def _on_record_button(self, is_on):
        self.record_on = is_on

    def _on_screenshot_btn(self):
        dt = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        save_dir = self.screenshot_save_path / dt
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = save_dir / "screenshot"
        height = self.window.size.height
        width = self.widget3d_width
        app = o3d.visualization.gui.Application.instance
        img = np.asarray(app.render_to_image(self.widget3d.scene, width, height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(f"{filename}-gui.png", img)
        if self.render_img is not None:
            img = np.asarray(self.render_img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(f"{filename}.png", img)

    def init_camera(self):
        height, width = int(self.window.size.height), int(self.widget3d_width)
        intrinsic = create_camera_intrinsic_from_size(
            width, height, self.fov[0], self.fov[1]
        )
        self.extrinsic = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(intrinsic, self.extrinsic, height, width, bounds)

    def update_view_candidates(self, xyz, direction):
        if self.show_view_candidates:
            self.widget3d.scene.remove_geometry("view_candidates_xyz")
            self.widget3d.scene.remove_geometry("view_candidates_direction")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(np.array([[1, 0, 0]]), (len(xyz), 1))
            )

            line_set = o3d.geometry.LineSet()
            line_points = np.concatenate([xyz, xyz + direction * 0.2], axis=0)
            indices = np.arange(len(xyz))
            lines = np.column_stack((indices, indices + len(xyz)))
            colors = np.tile(np.array([[0, 1, 0]]), (len(lines), 1))

            line_set.points = o3d.utility.Vector3dVector(line_points)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(colors)
            self.widget3d.scene.add_geometry("view_candidates_xyz", pcd, self.lit_dot)
            self.widget3d.scene.add_geometry(
                "view_candidates_direction", line_set, self.lit_line
            )
        else:
            self.widget3d.scene.remove_geometry("view_candidates_xyz")
            self.widget3d.scene.remove_geometry("view_candidates_direction")
            self.widget3d.scene.show_geometry("view_candidates_xyz", False)
            self.widget3d.scene.show_geometry("view_candidates_direction", False)

    def update_pcd(self):
        if not self.require_rasterization:
            self.widget3d.scene.remove_geometry("pcd")
            pcd = o3d.geometry.PointCloud()
            points = self.gaussian_cur.means.cpu().numpy()
            colors = self.gaussian_cur.harmonics.cpu().numpy().squeeze(1)
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            self.widget3d.scene.add_geometry("pcd", pcd, self.lit)
        else:
            self.widget3d.scene.show_geometry("pcd", False)

    def update_mesh(self):
        self.widget3d.scene.add_geometry("mesh", self.mesh_cur, self.mesh_render)

    def update_voxel(self):
        if self.voxel_type != "none":
            if self.voxel_type == "occ":
                vis_mask = self.voxel_cur.occ_mask
            elif self.voxel_type == "free":
                vis_mask = self.voxel_cur.free_mask
            elif self.voxel_type == "unknown":
                vis_mask = self.voxel_cur.unknown_mask
            elif self.voxel_type == "planning":
                vis_mask = self.voxel_cur.planning_mask
            elif self.voxel_type == "unexplored":
                vis_mask = self.voxel_cur.unexplored_mask
            elif self.voxel_type == "frontier":
                vis_mask = self.voxel_cur.frontier_mask
            elif self.voxel_type == "roi":
                vis_mask = self.voxel_cur.roi_mask

            if np.sum(vis_mask) == 0:
                self.widget3d.scene.remove_geometry("voxel")
            else:
                self.widget3d.scene.remove_geometry("voxel")
                voxel_line_set = create_voxel(
                    self.voxel_cur.voxel_centers[vis_mask],
                    self.voxel_cur.voxel_size,
                )
                self.widget3d.scene.add_geometry("voxel", voxel_line_set, self.lit_vox)
        else:
            self.widget3d.scene.remove_geometry("voxel")

    def update_path(self, xyz, color=[1, 0, 0]):
        if len(self.cam_path) > 1:
            if not (self.cam_path[-1] == xyz).all():
                self.widget3d.scene.remove_geometry("camera_path")
                self.cam_path.append(xyz)
                lines = []
                for i in range(len(self.cam_path) - 1):
                    lines.append([i, i + 1])

                colors = [color for i in range(len(lines))]
                canonical_line_set = o3d.geometry.LineSet()
                canonical_line_set.points = o3d.utility.Vector3dVector(self.cam_path)
                canonical_line_set.lines = o3d.utility.Vector2iVector(lines)
                canonical_line_set.colors = o3d.utility.Vector3dVector(colors)
                self.widget3d.scene.add_geometry(
                    "camera_path", canonical_line_set, self.lit_line
                )
        else:
            self.cam_path.append(xyz)

    def add_camera(self, camera, name, color=[0, 1, 0], size=0.1):
        if camera.rgb is not None:
            camera_rgb = o3d.geometry.Image(camera.rgb)
            self.in_rgb_widget.update_image(camera_rgb)
        else:
            camera_rgb = None

        if camera.depth is not None:
            camera_depth = o3d.geometry.Image(camera.depth)
            self.in_depth_widget.update_image(camera_depth)
        else:
            camera_depth = None

        C2W = camera.extrinsic.numpy()
        frustum = create_frustum(C2W, color, size=size)
        self.update_path(C2W[:3, 3])
        if name not in self.frame_dict.keys():
            self.combo_kf.add_item(name)
            self.frame_dict[name] = {
                "frustum": frustum,
                "rgb": camera_rgb,
                "depth": camera_depth,
            }
            self.widget3d.scene.add_geometry(
                f"{name}_camera", frustum.line_set, self.lit_line
            )
        frustum.update_pose(C2W)
        self.widget3d.scene.set_geometry_transform(
            f"{name}_camera", C2W.astype(np.float64)
        )
        self.widget3d.scene.show_geometry(f"{name}_camera", self.cameras_chbox.checked)
        self.widget3d.scene.show_geometry(f"camera_path", self.path_chbox.checked)

        return frustum

    def receive_mapper_data(self, q):
        if q is None:
            return
        mapper_packet = get_latest_queue(q)
        if mapper_packet is None:
            return

        # gaussian map update
        if mapper_packet.has_gaussians:
            self.gaussian_cur = mapper_packet.gaussian_packet
            self.output_info.text = "Number of Gaussians: {}".format(
                self.gaussian_cur.means.shape[0]
            )
            self.update_pcd()
            self.init = True

        # voxel map update
        if mapper_packet.has_voxels:
            self.voxel_cur = mapper_packet.voxel_packet
            self.update_voxel()

        # mesh update
        if mapper_packet.has_mesh:
            self.mesh_cur = o3d.geometry.TriangleMesh()
            self.mesh_cur.vertices = o3d.utility.Vector3dVector(
                mapper_packet.mesh_vertices
            )
            self.mesh_cur.triangles = o3d.utility.Vector3iVector(
                mapper_packet.mesh_triangles
            )
            self.mesh_cur.compute_vertex_normals()

        # camera frame update
        if mapper_packet.has_frame:
            current_frame = mapper_packet.current_frame

            if current_frame.id is None:
                frustum = self.add_camera(
                    current_frame, name="current", color=[0, 0, 1]
                )
            else:  # key frames
                name = "frame_{}".format(current_frame.id)
                frustum = self.add_camera(current_frame, name=name, color=[0, 1, 0])

            if self.followcam_chbox.checked:
                viewpoint = (
                    frustum.view_dir_behind
                    if self.staybehind_chbox.checked
                    else frustum.view_dir
                )
                self.widget3d.look_at(viewpoint[0], viewpoint[1], viewpoint[2])

    def receive_planner_data(self, q):
        if q is None:
            return
        planner_packet = get_latest_queue(q)
        if planner_packet is None:
            return
        if planner_packet.view_xyz is not None:
            self.update_view_candidates(
                planner_packet.view_xyz, planner_packet.view_direction
            )

    # camera for rendering: extrinsic is c2w in opencv format
    def get_current_cam(self):
        c2w_gl = self.widget3d.scene.camera.get_model_matrix()
        c2w = c2w_gl @ gl_cv
        image_gui = torch.zeros(
            (1, int(self.window.size.height), int(self.widget3d_width))
        )

        # normalized intrinsics
        _, H, W = image_gui.shape
        FoVx = np.deg2rad(self.fov[1])
        FoVy = np.deg2rad(self.fov[0])
        fx = fov2focal(FoVx, W) / W
        fy = fov2focal(FoVy, H) / H
        cx = 0.5
        cy = 0.5
        extrinsic = torch.from_numpy(c2w).float()
        intrinsic = torch.tensor(
            [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
        ).float()
        current_cam = Camera.init_from_gui(
            -1, extrinsic, intrinsic, H=H, W=W, fovx=FoVx, fovy=FoVy
        )
        return current_cam

    @torch.no_grad
    def rasterise(self, current_cam):
        extrinsics = current_cam.extrinsic.unsqueeze(0).to(self.device)
        intrinsics = current_cam.intrinsic.unsqueeze(0).to(self.device)
        render_h = int(self.ratio_slider.double_value * self.render_h)
        render_w = int(self.ratio_slider.double_value * self.render_w)

        remove_mask = self.gaussian_cur.confidences > self.thres_slider.double_value
        scale_factor = self.scale_slider.double_value

        gaussian_attr = (
            self.gaussian_cur.means[~remove_mask].to(self.device),
            self.gaussian_cur.harmonics[~remove_mask].to(self.device),
            self.gaussian_cur.opacities[~remove_mask].to(self.device),
            self.gaussian_cur.confidences[~remove_mask].to(self.device),
            (scale_factor * self.gaussian_cur.scales[~remove_mask]).to(self.device),
            self.gaussian_cur.rotations[~remove_mask].to(self.device),
        )
        background_color = torch.tensor(self.background, device=self.device)
        (rgb, depth, normal, opacity, d2n, confidence, importance, count, _) = (
            GaussianRenderer(
                extrinsics,
                intrinsics,
                gaussian_attr,
                background_color,
                (self.near, self.far),
                (render_h, render_w),
                self.device,
            ).render_view(front_only=self.frontonly_chbox.checked)
        )
        return {
            "rgb": rgb,
            "depth": depth,
            "confidence": confidence,
            "normal": normal,
            "d2n": d2n,
            "opacity": opacity,
        }

    def render_o3d_image(self, results, current_cam):
        if self.depth_chbox.checked:
            depth = results["depth"]
            depth = depth[0, :, :].detach().cpu().numpy()
            # max_depth = np.max(depth)
            depth = imgviz.depth2rgb(
                depth, min_value=self.near, max_value=self.far, colormap="jet"
            )
            depth = torch.from_numpy(depth)
            depth = torch.permute(depth, (2, 0, 1)).float()
            depth = (depth).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(depth)

        elif self.confidence_chbox.checked:
            confidence = results["confidence"]
            confidence = confidence[0, :, :].cpu().numpy()
            confidence = imgviz.depth2rgb(
                1 - confidence, min_value=0, max_value=1, colormap="jet"
            )
            confidence = torch.from_numpy(confidence)
            confidence = torch.permute(confidence, (2, 0, 1)).float()
            confidence = (confidence).byte().permute(1, 2, 0).contiguous().cpu().numpy()

            render_img = o3d.geometry.Image(confidence)

        elif self.opacity_chbox.checked:
            opacity = results["opacity"]
            opacity = opacity[0, :, :].detach().cpu().numpy()
            opacity = imgviz.depth2rgb(
                opacity, min_value=0.0, max_value=1.0, colormap="jet"
            )
            opacity = torch.from_numpy(opacity)
            opacity = torch.permute(opacity, (2, 0, 1)).float()
            opacity = (opacity).byte().permute(1, 2, 0).contiguous().cpu().numpy()
            render_img = o3d.geometry.Image(opacity)

        elif self.normal_chbox.checked:
            normal = results["normal"]
            normal = torch.nn.functional.normalize(normal, dim=0)
            normal = 1 - torch.add(normal, 1.00000) / 2

            normal = (
                (torch.clamp(normal, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )

            render_img = o3d.geometry.Image(normal)

        elif self.d2n_chbox.checked:
            normal = results["d2n"]
            normal = torch.nn.functional.normalize(normal, dim=0)
            normal = 1 - torch.add(normal, 1.00000) / 2

            normal = (
                (torch.clamp(normal, min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )

            render_img = o3d.geometry.Image(normal)

        else:
            rgb = (
                (torch.clamp(results["rgb"], min=0, max=1.0) * 255)
                .byte()
                .permute(1, 2, 0)
                .contiguous()
                .cpu()
                .numpy()
            )
            render_img = o3d.geometry.Image(rgb)
        return render_img

    def render_gui(self):
        if not self.init:
            return
        current_cam = self.get_current_cam()

        if self.require_rasterization and self.gaussian_cur is not None:
            results = self.rasterise(current_cam)
            self.render_img = self.render_o3d_image(results, current_cam)
            self.widget3d.scene.set_background(self.background, self.render_img)

    def scene_update(self):
        if self.q_mapper2gui is not None:
            self.receive_mapper_data(self.q_mapper2gui)
        if self.q_planner2gui is not None:
            self.receive_planner_data(self.q_planner2gui)
        self.render_gui()

    # camera for open3d gui: extrinsic is w2c in opencv format
    @property
    def gui_camera_params(self):
        view_matrix = np.asarray(self.widget3d.scene.camera.get_view_matrix())
        extrinsic = cv_gl @ view_matrix
        height, width = int(self.window.size.height), int(self.widget3d_width)
        intrinsic = create_camera_intrinsic_from_size(
            width, height, self.fov[0], self.fov[1]
        )
        resolution = [height, width]
        return extrinsic, intrinsic, resolution

    def record(self):
        # record camera path
        extrinsic, intrinsic, resolution = self.gui_camera_params
        if np.any(extrinsic != self.extrinsic):  # new camera extrinsic
            self.extrinsic = extrinsic
            save_list = (
                list(extrinsic.flatten()) + list(intrinsic.flatten()) + resolution
            )

            saved_path_name = "saved_path_{}.txt".format(
                self.combo_camera_path.selected_text
            )
            saved_path_file = os.path.join(self.path_save_path, saved_path_name)

            mode = "a" if os.path.exists(saved_path_file) else "w"
            with open(saved_path_file, mode) as f:
                f.write(" ".join(map(str, save_list)) + "\n")

    def _update_thread(self):
        while True:
            time.sleep(0.01)
            self.step += 1
            if self.process_finished:
                o3d.visualization.gui.Application.instance.quit()
                # Log("Closing Visualization", tag="GUI")
                break

            def update():
                if self.step % 3 == 0:
                    self.scene_update()

                if self.step % 100 and self.record_on:
                    self.record()

                if self.step >= 1e9:
                    self.step = 0

            gui.Application.instance.post_to_main_thread(self.window, update)


def run(init_event=None, cfg=None, params_gui=None):
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    win = GUI(cfg, params_gui)
    if init_event is not None:
        init_event.set()
    print("\n ----------load gui----------")
    app.run()
