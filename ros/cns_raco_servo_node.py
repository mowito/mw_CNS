"""
cns_raco_servo_node -- ROS 2 node running CNS (Correspondence-based Novel-view
Servoing) to publish camera velocity commands from RaCo+ALIKED keypoints.

Subscribes:
  - raco_features                    (vision_interfaces/msg/AlikedFeatures)
      Published by Raco_node/aliked_node.py ("raco_node"): RaCo keypoints +
      ALIKED descriptors extracted from the live camera stream. No raw
      images are consumed here -- detection/description already happened
      upstream, this node only matches and servos.
  - camera/camera/color/camera_info  (sensor_msgs/msg/CameraInfo)
      Read once to build the pinhole intrinsic used to project pixel
      keypoints onto the normalized camera plane CNS expects.

Publishes:
  - cam_vel  (geometry_msgs/msg/TwistStamped)
      Camera velocity [vx, vy, vz, wx, wy, wz] in the camera frame, as
      predicted by the CNS graph-VS network (checkpoints/cns.pth). Stamped
      with header.frame_id = the `camera_frame` parameter (default
      "camera_color_optical_frame") so downstream consumers (e.g. a
      statemachine_utils twistTransformer node) can retarget it into
      another frame via TF without guessing which frame it came from.

  - keypoint_error  (std_msgs/msg/Float32)
      Mean keypoint error (normalized camera-plane units) between the
      current and target correspondences, via the same witness-weighted,
      percentile-filtered metric cns.benchmark.stop_policy.PixelStopPolicy
      uses for offline evaluation. Published every processed frame; +inf
      whenever there is no usable correspondence (matching failed, too few
      keypoints, no target set yet). Intended as the convergence signal for
      servo state machines -- a velocity-magnitude threshold was
      deliberately not used here since it conflates controller gain/tuning
      with actual positioning error and is hard to compare across runs.

  - cns_viz/keypoints, cns_viz/graph  (sensor_msgs/msg/Image, bgr8)
      Only published when the `viz` parameter is true. Synthetic canvases
      (no raw camera frame is available here) showing the target/current
      keypoints and the CNS clustering graph, via cns.utils.visualize --
      view with `ros2 run rqt_image_view rqt_image_view`.

Services:
  - ~/set_target  (std_srvs/srv/Trigger)
      Freezes the *next* incoming AlikedFeatures message as the new servo
      target. The first message received after startup is captured as the
      initial target automatically, no call needed to get going.

Matching between the frozen target frame and each incoming current frame is
done with LightGlue(features="raco-aliked") directly on the keypoints and
descriptors raco_node already extracted -- this is the same pairing
documented in Raco_node/aliked_node.py's own docstring.

Run with a Python that has both ROS 2 (rclpy, the vision_interfaces message
package on PYTHONPATH -- source lightglue_visual_servoing/install/setup.bash)
and CNS's deps (torch, torch_geometric, scikit-learn) importable, e.g.:

    source /opt/ros/humble/setup.bash
    source ~/lightglue_visual_servoing/install/setup.bash
    python3 ros/cns_raco_servo_node.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.executors import ExternalShutdownException
from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import Twist, TwistStamped
from std_msgs.msg import Float32
from std_srvs.srv import Trigger
from cv_bridge import CvBridge

_THIS_DIR = Path(__file__).resolve().parent
_MW_CNS_ROOT = _THIS_DIR.parent
_LIGHTGLUE_ROOT = Path.home() / "lightglue_visual_servoing" / "LightGlue"

for _p in (_MW_CNS_ROOT, _LIGHTGLUE_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from lightglue import LightGlue  # noqa: E402

from cns.utils.perception import CameraIntrinsic  # noqa: E402
from cns.frontend.utils import Correspondence  # noqa: E402
from cns.midend.corr2graph import Midend  # noqa: E402
from cns.benchmark.controller import GraphVSController  # noqa: E402
from cns.benchmark.stop_policy import PixelStopPolicy  # noqa: E402
from cns.utils.visualize import draw_keypoints, draw_graph  # noqa: E402


# Correspondence carries tar_img/cur_img only for optional visualization and
# an SSIM stop-criterion, neither of which this node uses -- a shared 1x1
# placeholder avoids a full-frame allocation on every callback.
_DUMMY_IMG = np.zeros((1, 1, 3), dtype=np.uint8)


class RacoLightGlueFrontend:
    """
    Matches already-extracted RaCo+ALIKED keypoints/descriptors between a
    frozen target frame and each incoming current frame with
    LightGlue(features="raco-aliked"), and builds the same Correspondence
    contract cns.frontend.classic.Classic / cns.frontend.superglue.SuperGlue
    produce from raw images -- except here detection+description already
    happened upstream in raco_node, so this only matches.
    """

    def __init__(self, device, ransac=True, min_matches=8):
        self.device = torch.device(device)
        self.ransac = ransac
        self.min_matches = min_matches
        self.matcher = LightGlue(features="raco-aliked").eval().to(self.device)

        self.tar_pos = None
        self.tar_des = None
        self.tar_wh = None
        self.tar_img_changed = True

    def _feat_dict(self, kpts, des, wh):
        return {
            "keypoints": torch.from_numpy(kpts).float().unsqueeze(0).to(self.device),
            "descriptors": torch.from_numpy(des).float().unsqueeze(0).to(self.device),
            "image_size": torch.tensor([wh], dtype=torch.float32, device=self.device),
        }

    def update_target(self, kpts, des, width, height):
        if len(kpts) < self.min_matches:
            return False
        self.tar_pos = kpts.astype(np.float32)
        self.tar_des = des.astype(np.float32)
        self.tar_wh = (float(width), float(height))
        self.tar_img_changed = True
        return True

    @torch.no_grad()
    def process_current(self, kpts, des, width, height, intrinsic: CameraIntrinsic):
        if self.tar_pos is None or len(kpts) < self.min_matches:
            return None

        data = {
            "image0": self._feat_dict(self.tar_pos, self.tar_des, self.tar_wh),
            "image1": self._feat_dict(kpts, des, (float(width), float(height))),
        }
        # matches: (M, 2) int64, columns = [index into tar_pos, index into kpts]
        matches = self.matcher(data)["matches"][0].cpu().numpy()
        if len(matches) < self.min_matches:
            return None

        if self.ransac:
            tar_pts = self.tar_pos[matches[:, 0]]
            cur_pts = kpts[matches[:, 1]]
            if len(matches) == 4:
                _, mask = cv2.findHomography(tar_pts, cur_pts, cv2.RANSAC, 5.0)
            else:
                _, mask = cv2.findEssentialMat(
                    tar_pts, cur_pts, intrinsic.K, cv2.RANSAC, 0.999, 3.0)
            if mask is None:
                return None
            matches = matches[mask.ravel().astype(bool)]
            if len(matches) < self.min_matches:
                return None

        cur_pos_aligned = np.zeros_like(self.tar_pos)
        valid_mask = np.zeros(len(self.tar_pos), dtype=bool)
        cur_pos_aligned[matches[:, 0]] = kpts[matches[:, 1]]
        valid_mask[matches[:, 0]] = True

        corr = Correspondence(
            intrinsic=intrinsic,
            tar_img=_DUMMY_IMG, tar_pos=self.tar_pos,
            cur_img=_DUMMY_IMG, cur_pos=kpts,
            match=matches, valid_mask=valid_mask, cur_pos_aligned=cur_pos_aligned,
            detector_name="RaCo+ALIKED+LightGlue",
            tar_img_changed=self.tar_img_changed,
        )
        self.tar_img_changed = False
        return corr


class CnsRacoServoNode(Node):
    def __init__(self):
        super().__init__("cns_raco_servo_node")

        self.declare_parameter("features_topic", "raco_features")
        self.declare_parameter("camera_info_topic", "camera/camera/color/camera_info")
        self.declare_parameter("cmd_vel_topic", "cam_vel")
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("keypoint_error_topic", "keypoint_error")
        # Best checkpoint from the 08_05_01_05_27_CNS_adaptive_gain_long
        # training run -- GraphVS trained against the adaptive-gain PBVS
        # supervisor (cns/sim/supervisor.py:pbvs) instead of the original
        # fixed-gain targets. GraphVSController loads this full Trainer
        # checkpoint dict ({"net": GraphVS, ...}) directly, no state-dict
        # extraction needed.
        self.declare_parameter(
            "checkpoint_path", str(
                _MW_CNS_ROOT / "checkpoints" / "08_05_01_05_27_CNS_adaptive_gain_long"
                / "checkpoint_best.pth"))
        self.declare_parameter("device", "cuda:0" if torch.cuda.is_available() else "cpu")
        # Distance (metres) from the camera to the scene centre at the
        # target pose -- CNS's `tPo_norm` scale factor, which directly
        # scales the predicted linear velocity. Re-read every callback so
        # it can be tuned live with `ros2 param set`.
        self.declare_parameter("scene_scale", 1.0)
        self.declare_parameter("ransac", True)
        self.declare_parameter("min_matches", 8)
        # Off by default -- draw_keypoints/draw_graph run clustering-derived
        # drawing every frame, pure added cost with no consumer if nobody's
        # watching cns_viz/*. Re-read live so it can be toggled without a
        # restart, same as scene_scale.
        self.declare_parameter("viz", False)

        device = self.get_parameter("device").value
        checkpoint_path = self.get_parameter("checkpoint_path").value
        ransac = self.get_parameter("ransac").value
        min_matches = self.get_parameter("min_matches").value

        self.get_logger().info(
            f"Loading CNS controller from {checkpoint_path} on {device} ...")
        self.controller = GraphVSController(checkpoint_path, device)
        self.frontend = RacoLightGlueFrontend(device, ransac, min_matches)
        self.midend = Midend()
        # waiting_time/conduct_thresh only matter for the hysteresis in
        # PixelStopPolicy.__call__, which this node never uses -- only
        # .calculate_error(data) is called, to get a single repeatable
        # error number per frame and let the state machine own the
        # convergence threshold/hysteresis.
        self.stop_policy = PixelStopPolicy(waiting_time=0.0, conduct_thresh=0.0)

        self.intrinsic = None
        self._target_pending = True   # next incoming frame becomes the target
        self._new_scene = True        # next graph gets its RNN hidden state reset

        from vision_interfaces.msg import AlikedFeatures

        sub_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self.create_subscription(
            AlikedFeatures, self.get_parameter("features_topic").value,
            self._features_callback, sub_qos)
        self.create_subscription(
            CameraInfo, self.get_parameter("camera_info_topic").value,
            self._camera_info_callback, 1)
        self.vel_pub = self.create_publisher(
            TwistStamped, self.get_parameter("cmd_vel_topic").value, 10)
        self.keypoint_error_pub = self.create_publisher(
            Float32, self.get_parameter("keypoint_error_topic").value, 10)
        self.create_service(Trigger, "~/set_target", self._set_target_cb)

        self.bridge = CvBridge()
        self.viz_keypoints_pub = self.create_publisher(Image, "cns_viz/keypoints", 1)
        self.viz_graph_pub = self.create_publisher(Image, "cns_viz/graph", 1)

        self.get_logger().info("cns_raco_servo_node ready, waiting for features/camera_info.")

    def _camera_info_callback(self, msg: CameraInfo):
        self.intrinsic = CameraIntrinsic(
            width=msg.width, height=msg.height,
            fx=msg.k[0], fy=msg.k[4], cx=msg.k[2], cy=msg.k[5],
        )

    def _set_target_cb(self, request, response):
        self._target_pending = True
        response.success = True
        response.message = "Next incoming frame will be captured as the new target."
        return response

    def _stamped(self, twist: Twist) -> TwistStamped:
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_parameter("camera_frame").value
        msg.twist = twist
        return msg

    def _publish_keypoint_error(self, error: float):
        msg = Float32()
        msg.data = error
        self.keypoint_error_pub.publish(msg)

    def _publish_stop(self):
        self.vel_pub.publish(self._stamped(Twist()))
        self._publish_keypoint_error(float("inf"))
        self.stop_policy.reset()

    def _features_callback(self, msg):
        n = msg.num_features
        if n == 0:
            self._publish_stop()
            return

        kpts = np.frombuffer(msg.keypoints, dtype=np.float32).reshape(n, 2).copy()
        des = np.frombuffer(msg.descriptors, dtype=np.float32).reshape(n, -1).copy()

        if self._target_pending:
            if self.frontend.update_target(kpts, des, msg.image_width, msg.image_height):
                self._target_pending = False
                self._new_scene = True
                self.get_logger().info(f"Captured new target frame ({n} keypoints).")
            else:
                self.get_logger().warn(f"Too few keypoints ({n}) to set target, retrying.")
            return

        if self.intrinsic is None:
            self.get_logger().warn(
                "No CameraInfo received yet, cannot convert pixels to the "
                "normalized camera plane.", throttle_duration_sec=5.0)
            return

        corr = self.frontend.process_current(
            kpts, des, msg.image_width, msg.image_height, self.intrinsic)
        if corr is None:
            self._publish_stop()
            return

        data = self.midend.get_graph_data(corr)
        if self._new_scene:
            data.start_new_scene()
            self._new_scene = False
        data.set_distance_scale(self.get_parameter("scene_scale").value)
        # only consumed by draw_keypoints/draw_graph below, harmless to set
        # unconditionally (matches CorrespondenceBasedPipeline's own pattern)
        setattr(data, "intrinsic", self.intrinsic)

        if self.get_parameter("viz").value:
            self._publish_viz(data, msg.header)

        self._publish_keypoint_error(float(self.stop_policy.calculate_error(data)))

        vel = self.controller(data)

        twist = Twist()
        twist.linear.x, twist.linear.y, twist.linear.z = (float(v) for v in vel[:3])
        twist.angular.x, twist.angular.y, twist.angular.z = (float(v) for v in vel[3:])
        self.vel_pub.publish(self._stamped(twist))

    def _publish_viz(self, data, header):
        kp_img = draw_keypoints(data)
        graph_img = draw_graph(data)
        for pub, img in ((self.viz_keypoints_pub, kp_img), (self.viz_graph_pub, graph_img)):
            img_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            img_msg.header = header
            pub.publish(img_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CnsRacoServoNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
