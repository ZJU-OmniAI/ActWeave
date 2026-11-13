import base64
from dataclasses import dataclass, field, fields
import inspect
import io
import math
import sys
from typing import Literal

import numpy as np
from PIL import Image
from spatial import SpatialAPI, Frame, Geometry, FrameImage, BoundFeature
from manipulation import ManipulationAPI


@dataclass(frozen=True)
class ArmState:
    """Public arm state: TCP/link6 positions in world metres, quaternion w,x,y,z, Euler degrees and six joint radians. gripper is opening readback; requested_opening is the held command."""
    tcp_pos: list[float]
    eef_pos: list[float]
    eef_quat: list[float]
    pitch: float
    yaw: float
    roll: float
    gripper: float
    requested_opening: float | None
    joints: list[float]


@dataclass(frozen=True)
class State:
    """Public robot snapshot at step; step_limit and truncated describe the native action allowance. left/right contain ArmState."""
    step: int
    step_limit: int
    task: str
    left: ArmState
    right: ArmState
    truncated: bool
    ended: bool

    def arm(self, name: str) -> ArmState:
        """Select left or right without copying the snapshot."""
        if name not in ("left", "right"):
            raise ValueError("arm must be left or right")
        return getattr(self, name)


def _state(value):
    return State(**{**value, "left": ArmState(**value["left"]), "right": ArmState(**value["right"])})


@dataclass(frozen=True)
class Budget:
    """Remaining wall/cell seconds and request, upstream-attempt, cell and native-action counts at sampled_at. None request/attempt limits mean unlimited when the corresponding used count is available; missing upstream readings remain None."""
    sampled_at: str
    wall_seconds_remaining: float
    cell_seconds_remaining: float | None
    model_requests_used: int
    model_request_limit: int | None
    upstream_attempts_used: int | None
    upstream_attempt_limit: int | None
    upstream_attempts_recorded_at: str | None
    upstream_attempts_note: str | None
    jev_requests_used: int | None
    jev_upstream_attempts_used: int | None
    cells_used: int
    cell_limit: int
    native_steps_used: int
    native_step_limit: int


@dataclass(frozen=True)
class Move:
    """Actual TCP feedback. reached requires completed joint tracking plus <=2 mm/1 deg target residuals, not task success. Each waypoint is commanded at least once; tracking tolerance is min(0.001 rad, 10% of its initial joint error), floored at 1e-8 rad. plan contains solver results; state and residuals describe actual motion. reason identifies solver/tracking/runtime state, not object contact or task failure. start_step/end_step delimit the public observations around the call. motion_feedback reports both arms' commanded/measured joints and TCP deltas; limit_kind distinguishes caller_max_steps from native_action_limit on a step_budget return. tracking reports the active waypoint index, action count, joint residual and tolerance, without a stall judgement."""
    start_step: int
    end_step: int
    arm: str
    target_xyz: list[float]
    target_pose: list[list[float]]
    start_xyz: list[float]
    final_xyz: list[float]
    dist_to_target_m: float
    orientation_error_deg: float
    pitch: float
    yaw: float
    roll: float
    reached: bool
    path: str | None
    ik_error: str | None
    reason: str | None
    steps_used: int
    max_joint_error_rad: float | None
    waypoints_completed: int
    path_complete: bool
    tracking: dict | None
    motion_feedback: dict
    plan: dict
    state: State
    truncated: bool


@dataclass(frozen=True)
class Gripper:
    """Opening feedback: 1 open, 0 empty close. holding is a close-width heuristic; width_m=0.0023+0.1091*opening estimates jaw gap in metres for either command. steps_used and start_step/end_step/step refer to native actions and observations. motion_feedback exposes both arms' command tracking and displacement."""
    start_step: int
    end_step: int
    arm: str
    commanded: str
    opening: float
    holding: bool
    width_m: float
    stable: bool
    motion_feedback: dict
    steps_used: int
    step: int
    step_limit: int
    truncated: bool


@dataclass(frozen=True)
class Home:
    """Actual return feedback: reached means a return command executed and selected joints are within 0.001 rad of initial angles. lifted_m/lift_plans report an explicitly requested preliminary rise; steps_used includes all actions. state is actual public feedback; motion_feedback includes both arms and the step-limit source."""
    start_step: int
    end_step: int
    arm: str
    reached: bool
    max_joint_error_rad: float
    steps_used: int
    truncated: bool
    lifted_m: dict[str, float]
    lift_plans: dict
    motion_feedback: dict
    state: State
    reason: str | None


@dataclass(frozen=True)
class Region:
    """Mask candidate: bool mask, centroid [row,col], box [left,top,right,bottom], area_px and segmentation score."""
    centroid: list[int]
    box: list[int]
    area_px: int
    score: float | None
    mask: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class Segment:
    """Current-camera candidates. Top-level fields describe the selected highest-score region; regions retains the masks."""
    found: bool
    camera: str
    step: int
    centroid: list[int] | None
    area_px: int
    score: float | None
    box: list[int] | None
    regions: list[Region]


@dataclass(frozen=True)
class ObjectSight:
    """CAP estimate above support_z in world metres. center_xyz is fitted footprint centre at half height; length_m >= width_m. long_axis_deg/grasp_yaw are headings from +x. found refers to the mask; geometry can remain None. geometry labels multi_view_columns, single_view_guess or retained_height; height_memory is used/rejected/None. fit is silhouette agreement. view_support retains full masks, fitted pixels and calibration in Python; repr shows compact matching evidence without those arrays. surface_xyz is the point-anchor intersection."""
    found: bool
    camera: str
    step: int
    center_xyz: list[float] | None
    top_z: float | None
    height_m: float | None
    length_m: float | None
    width_m: float
    long_axis_deg: float | None
    grasp_yaw: float | None
    score: float | None
    centroid: list[int] | None
    area_px: int
    views: int
    note: str | None
    view_support: list[dict]
    height_memory: str | None
    fit: float | None
    geometry: str | None
    shape: str | None

    source: str
    unit: str
    support_z: float
    assumptions: dict
    surface_xyz: list[float] | None

    def __repr__(self):
        values = {item.name: getattr(self, item.name) for item in fields(self)}
        values['view_support'] = [{key: value for key, value in view.items() if key not in {'mask', 'fit_mask_pixels', 'intrinsics', 'camera_to_world'}} for view in self.view_support]
        return type(self).__name__ + '(' + ', '.join(f'{key}={value!r}' for key, value in values.items()) + ')'


@dataclass(frozen=True)
class SeenPoint:
    """CAP pixel [row,col] intersection: world xyz metres and optical depth_m. surface is object/table/unresolved; table denotes an explicit support-plane estimate. views/note describe the fit."""
    pixel: list[int]
    xyz: list[float] | None
    depth_m: float | None
    camera: str
    surface: str
    views: int
    note: str | None


def _number(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("numeric arguments must be finite")
    return 0.0 if result == 0 else result


def _vector(value, size):
    result = [_number(x) for x in value]
    if len(result) != size:
        raise ValueError(f"expected {size} numbers")
    return result


def _rgb(data):
    return np.array(Image.open(io.BytesIO(base64.b64decode(data))).convert("RGB"))


class RobotAPI(ManipulationAPI):
    ARMS = ("left", "right")
    LEFT = "left"
    RIGHT = "right"
    OPEN = -1
    CLOSE = 1
    KEEP = 0
    CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")

    def __init__(self, tools, task):
        self._tools = tools
        self._task = task
        self._shown = []
        self._displayed = []
        self._objects = {}

    @property
    def task(self) -> str:
        return self._task

    async def recovery_note(self, text: str | None = None) -> dict | None:
        """Read or replace the episode recovery note. Returns text, recorded_at and recorded_step; survives Python restarts."""
        if text is not None and not isinstance(text, str):
            raise ValueError("text must be a string or None")
        return await self._tools.recovery_note({} if text is None else {"text": text})

    async def transition_record(self, id: str, description: dict | None = None, *, active: bool | None = None) -> dict:
        """Optionally create/replace an Astra description or set action association.
        description needs objects, basis, goal_delta, expected_delta, preserve,
        decision and anchors; optional watch needs Jev questions, with cameras optional.
        Optional basis_receipts references canonical execution/measurement ids.
        target_anchors retains original reference frames across revisions unless
        explicitly replaced ([] clears). depends_on=[{"id": ..., "revision": ...}]
        includes historical source evidence; omitted revision pins the latest
        existing revision now. The harness selects intermediate observation frames.
        Each replacement starts a new revision at the current step. No physics,
        consultation or stage decision. Host records survive Python restarts.
        """
        args = {"id": id}
        if description is not None:
            args["description"] = description
        if active is not None:
            args["active"] = active
        return await self._tools.transition_record(args)

    async def transition_read(self, id: str | None = None, *, revision: int | None = None, after_receipt: int = 0, after_observation: int = 0) -> dict:
        """Read original descriptions, action receipts and Jev advice; omit id for
        the catalog and episode evidence, including calls without a transition.
        Without id, after_observation selects consultation ids; with id it
        selects observation ids. Defaults to current revision; filters are explicit.
        Historical evidence does not establish the present physical state.
        """
        args = {"after_receipt": after_receipt, "after_observation": after_observation}
        if id is not None:
            args["id"] = id
        if revision is not None:
            args["revision"] = revision
        return await self._tools.transition_read(args)

    async def snapshot(self, *, step: int = -1, cameras: list[str] | None = None) -> dict:
        """Read same-step state and selected camera calibration/original PNGs.
        snapshots[0].cameras contains png_base64, camera and step per view.
        Returns data without displaying images; use show/display separately.
        """
        args = {"step": step, "display": False}
        if cameras is not None:
            args["cameras"] = cameras
        return await self._tools.episode_snapshot(args)

    async def operation_experience(self, id: str | None = None):
        """Read optional experience or list its catalog. Advice is not a required
        action recipe. Astra decides whether it applies to the current scene.
        """
        return await self._tools.operation_experience({} if id is None else {"id": id})

    async def budget(self) -> Budget:
        """Read fresh remaining time and execution allowances without advancing physics."""
        return Budget(**await self._tools.budget({}))

    async def state(self, step: int = -1) -> State:
        """Read a public snapshot, without stepping physics or attaching images.
        step=-1 selects current; nonnegative step reads the saved observation.
        """
        value = await self._tools.observation({"step": int(step)})
        return _state(value)

    async def image(self, camera: str = "cam_head", *, resolution: str = "high", step: int = -1) -> np.ndarray:
        """Get an RGB uint8 array, shape (480,640,3), for computations. No image
        enters model context. step=-1 is current, 0 initial; resolution="high" is the only mode. No resize or depth.
        """
        if resolution != "high":
            raise ValueError("the cameras provide only high resolution")
        value = await self._tools.image({"camera": str(camera), "step": int(step)})
        return _rgb(value["png_base64"])

    async def artifact(self, name: str, *, step: int = -1) -> bytes:
        """Get original PNG bytes: cam_head.png, cam_left_wrist.png or
        cam_right_wrist.png. Use image() for decoded arrays.
        """
        camera = name.removesuffix(".png")
        if name != camera + ".png" or camera not in self.CAMERAS:
            raise ValueError("artifact must name a public camera PNG")
        value = await self._tools.image({"camera": str(camera), "step": int(step)})
        return base64.b64decode(value["png_base64"])

    async def show(self, camera: str = "cam_head", *, step: int = -1) -> None:
        """Request a full RGB camera image for the next model decision.
        Requests for the same camera/step are combined. Images are read after
        the cell code stops; step=-1 then means its final public observation.
        Save image() during the cell to retain an intermediate frame instead.
        No physics advances; showing pictures consumes model tokens.
        """
        if camera not in self.CAMERAS:
            raise ValueError("camera must name a public RGB camera")
        request = (str(camera), int(step))
        if request not in self._shown:
            self._shown.append(request)

    async def _flush_images(self):
        shown, self._shown = self._shown, []
        displayed, self._displayed = self._displayed, []
        errors = []
        for camera, step in shown:
            try:
                await self._tools.show({"camera": camera, "step": step})
            except Exception as exc:
                errors.append(f"show({camera}, step={step}): {type(exc).__name__}: {exc}")
        for image in displayed:
            try:
                await display(image, self._tools)
            except Exception as exc:
                errors.append(f"display: {type(exc).__name__}: {exc}")
        return errors

    async def camera_meta(self, camera: str = "cam_head", *, step: int = -1) -> dict:
        """Return camera, resolved observation step, shape [480,640,3], intrinsic_matrix (3x3 pixels) and
        extrinsic_matrix (4x4 camera-to-world). Camera +x is right, +y up,
        viewing along -z; image rows increase down. Wrist poses follow public
        link6 pose; head is fixed. No hidden poses or native depth map are available.
        """
        return await self._tools.camera({"camera": str(camera), "step": int(step)})

    async def segment(self, text_prompt: str | None = None, camera: str = "cam_head", min_score: float = 0.2, *, point: list[int] | None = None, points: list[list[int]] | None = None, labels: list[int] | None = None, box: list[float] | None = None) -> Segment:
        """Segment exactly one of a nonempty text description, pixel point
        [row,col], multiple points with optional 0/1 labels (default all positive),
        or box [left,top,right,bottom]. Reads the current image. Negative points
        exclude regions selected by the caller; no identity is inferred.
        Centroids describe mask pixels, not verified object identity or depth.
        All mask arrays stay in Python until displayed. Multiple objects may
        match a phrase.
        """
        if point is not None and len(point) != 2:
            raise ValueError("point requires row and col")
        query = {k: v for k, v in {"text": str(text_prompt) if text_prompt is not None else None, "point": [int(v) for v in point] if point is not None else None, "points": points, "box": _vector(box, 4) if box is not None else None}.items() if v is not None}
        if len(query) != 1 or (text_prompt is not None and not text_prompt.strip()):
            raise ValueError("provide exactly one nonempty text_prompt, point, points or box")
        if labels is not None:
            query["labels"] = labels
        value = await self._tools.segment({"camera": camera, "min_score": _number(min_score), **query})
        regions = [Region(centroid=[int(x) for x in r["centroid"]], box=r["bbox"], area_px=r["area"], score=r["score"], mask=np.array(Image.open(io.BytesIO(base64.b64decode(r["mask"])))) > 0) for r in value["regions"]]
        regions.sort(key=lambda r: -r.score if r.score is not None else 0)
        first = regions[0] if regions and (len(regions) == 1 or regions[0].score is not None) else None
        return Segment(bool(regions), camera, value["step"], first.centroid if first else None, first.area_px if first else 0, first.score if first else None, first.box if first else None, regions)


    async def locate(self, text_prompt: str | None = None, camera: str = 'cam_head', min_score: float = 0.2, *, support_z: float = 0.765, point: list[int] | None = None, points: list[list[int]] | None = None, labels: list[int] | None = None, box: list[float] | None = None, view_prompts: dict | None = None, region_index: int = 0, masks: list[tuple[Segment, int]] | None = None) -> ObjectSight:
        """Fit CAP geometry to current RGB masks above support_z in world metres.
        Supply one text/point/points/box anchor; pixels are [row,col]. points
        accepts 0/1 labels, and view_prompts accepts the same prompt forms.
        Alternatively masks=[(Segment,region_index), ...] supplies explicitly
        selected, optionally edited boolean masks from one saved step, one per
        camera including camera. No SAM3 calls or automatic view selection run
        in mask mode. Mask pixels are preserved; matching identity is your
        assumption. Historical masks use their own calibration and robot state.
        Do not combine masks with prompts, view_prompts or region_index.
        view_prompts corrects other cameras with explicit prompts. region_index
        selects descending mask score. Example: await robo.locate(point=[240,320]).
        """
        if masks is not None:
            if any(v is not None for v in (text_prompt, point, points, labels, box, view_prompts)) or region_index != 0 or not masks:
                raise ValueError('mask mode requires nonempty selections without segmentation prompts')
            supplied = []
            for segment, index in masks:
                if not isinstance(segment, Segment) or isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(segment.regions):
                    raise ValueError('each selection requires Segment and a valid region index')
                pixels = segment.regions[index].mask
                if pixels.dtype != np.bool_ or pixels.ndim != 2 or not pixels.any():
                    raise ValueError('selected masks must be nonempty 2D boolean arrays')
                encoded = io.BytesIO()
                Image.fromarray(pixels).save(encoded, format='PNG')
                supplied.append({'camera': segment.camera, 'step': segment.step, 'png_base64': base64.b64encode(encoded.getvalue()).decode('ascii')})
            value = await self._tools.locate({'camera': camera, 'masks': supplied, 'support_z': _number(support_z)})
        else:
            prompt = {key: value for key, value in {'text': text_prompt, 'point': point, 'points': points, 'box': box}.items() if value is not None}
            if len(prompt) != 1 or (text_prompt is not None and not text_prompt.strip()):
                raise ValueError('provide exactly one nonempty text, point, points or box')
            if labels is not None:
                prompt['labels'] = labels
            prompt['min_score'] = _number(min_score)
            value = await self._tools.locate({'camera': camera, 'prompt': prompt, 'support_z': _number(support_z), 'view_prompts': view_prompts or {}, 'region_index': int(region_index)})
        for view in value['view_support']:
            if 'mask' in view:
                view['mask'] = np.array(Image.open(io.BytesIO(base64.b64decode(view['mask'])))) > 0
                view['fit_mask_pixels'] = np.asarray(view['fit_mask_pixels'], dtype=np.int32).reshape(-1, 2)
        return ObjectSight(**value)


    async def back_project(self, row: int, col: int, camera: str = 'cam_head', *, support_z: float = 0.765) -> SeenPoint:
        """Intersect the current pixel ray with a CAP-fitted surface.
        surface='table' labels the support_z plane estimate when no solid is fitted.
        """
        return SeenPoint(**await self._tools.back_project({'row': int(row), 'col': int(col), 'camera': camera, 'support_z': _number(support_z)}))


    async def ray_plane(self, row: float, col: float, z: float, camera: str = "cam_head") -> list[float]:
        """Intersect the current pixel ray with an explicitly chosen world-z
        plane in metres. This assumes the height you supply; it does not measure it.
        """
        value = await self._tools.ray_plane({"camera": str(camera), "row": _number(row), "col": _number(col), "z": _number(z)})
        return value["xyz"]

    async def move_to(self, xyz: list[float], arm: str = "right", gripper: int = 0, pitch: float | None = None, yaw: float | None = None, roll: float | None = None, path: Literal["auto", "line", "joint"] = "auto", *, max_steps: int | None = None, max_joint_step_rad: float | None = None) -> Move:
        """Move TCP to world xyz metres; Euler angles are degrees. No angles retains current rotation. With any angle supplied, omissions use current Euler values, except omitted roll=0 within one degree of vertical pitch. No automatic half-turn or alternate target. line requests straight TCP waypoints, joint interpolates endpoint IK, auto authorizes either. No collision veto or lack-of-progress stop. max_steps caps native actions; None uses the remaining native allowance. max_joint_step_rad optionally sets a commanded joint increment in (0,0.15] radians per native action; default 0.15, not Cartesian speed or force. Returns Move with solver results, actual state, residuals, waypoint progress and actual start_step/end_step."""
        if gripper not in (-1, 0, 1):
            raise ValueError("gripper must be OPEN, CLOSE or KEEP")
        value = await self._tools.move({"side": str(arm), "xyz": _vector(xyz, 3), "gripper": int(gripper), "path": path, **({"max_steps": int(max_steps)} if max_steps is not None else {}), **({"max_joint_step_rad": _number(max_joint_step_rad)} if max_joint_step_rad is not None else {}), **{k: _number(v) for k, v in {"pitch": pitch, "yaw": yaw, "roll": roll}.items() if v is not None}})
        value["state"] = _state(value["state"])
        return Move(**value)

    async def set_gripper(self, arm: str, gripper: int, opening: float | None = None, *, steps: int = 10) -> Gripper:
        """Run OPEN/CLOSE for steps native actions (default 10), without a stability-based stop. OPEN may specify opening in (0,1]; KEEP in later motion retains that command. stable reports only the last opening change below 0.004."""
        if gripper not in (-1, 1) or (opening is not None and (gripper != -1 or not 0 < opening <= 1)):
            raise ValueError("use OPEN/CLOSE; partial opening is only for OPEN")
        return Gripper(**await self._tools.gripper({"side": str(arm), "opening": (1 if opening is None else _number(opening)) if gripper == -1 else 0, "steps": steps}))

    async def wait(self, steps: int, *, capture_at: list[int] | None = None) -> dict:
        """Repeat the latest joint/opening command for steps native actions. capture_at contains relative offsets; returned frames are absolute saved steps. The result is a dict with state: State and motion_feedback for both arms."""
        if isinstance(steps, bool) or not isinstance(steps, (int, np.integer)) or steps < 0:
            raise ValueError("steps must be a nonnegative integer")
        offsets = list(capture_at) if capture_at is not None else []
        if any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 0 <= n <= steps for n in offsets):
            raise ValueError("capture_at must contain integer offsets in [0,steps]")
        value = await self._tools.wait({"steps": int(steps), "capture_at": [int(n) for n in offsets]})
        value["state"] = _state(value["state"])
        return value

    async def home(self, arm: str = 'both', *, max_steps: int | None = None, lift: bool = False, max_joint_step_rad: float | None = None) -> Home:
        """Return selected arms to initial joints, retaining opening commands.
        lift=False by default. Explicit True first requests a 6 cm straight rise
        per selected arm; a failed rise returns without the joint return. No
        alternate rise or hidden action cap. max_steps caps native actions; None uses the remaining native allowance.
        Optional max_joint_step_rad in (0,0.15] bounds each return command increment; omission retains the direct initial-joint command and 0.15-rad lift increments. Reports actual state, rise, joint error, solver results and start_step/end_step.
        """
        value = await self._tools.home({'side': arm, 'lift': bool(lift), **({'max_steps': int(max_steps)} if max_steps is not None else {}), **({'max_joint_step_rad': _number(max_joint_step_rad)} if max_joint_step_rad is not None else {})})
        value['steps_used'] = value.pop('steps')
        value['state'] = _state(value['state'])
        return Home(**value)


    async def joint_action(self, left: list[float], right: list[float], left_opening: float, right_opening: float) -> State:
        """Send one native action: six radians per arm, openings in [0,1]. No
        IK or path checks. Returns the next public state; consumes one action.
        """
        value = await self._tools.joint_action({"left": _vector(left, 6), "right": _vector(right, 6), "left_opening": _number(left_opening), "right_opening": _number(right_opening)})
        return _state(value)


async def display(image, tools):
    if isinstance(image, (bytes, bytearray)):
        encoded = bytes(image)
    else:
        data = io.BytesIO()
        if isinstance(image, Image.Image):
            picture = image
        else:
            pixels = np.asarray(image)
            if pixels.dtype != np.uint8:
                pixels = np.clip(pixels, 0, 255).astype(np.uint8)
            picture = Image.fromarray(pixels)
        picture.save(data, format="PNG")
        encoded = data.getvalue()
    await tools.display_image({"png_base64": base64.b64encode(encoded).decode("ascii")})


def reference():
    lines = ["# Executable Python interface", "Use `await` on every async method. `robo`, `np`, `math`, `time`, `log`, `display` and `finish` are predefined. Dataclass result fields use attribute access; dict results use keys. Only methods marked async use await. Do not mutate geometry/frame arrays; create a revised value instead. These classes are real runtime types; arrays and intermediate values stay in Python.", "robo.task: str (the native instruction)", "robo.ARMS = ('left', 'right'); robo.LEFT = 'left'; robo.RIGHT = 'right'", "robo.OPEN = -1; robo.CLOSE = 1; robo.KEEP = 0", "robo.CAMERAS = ('cam_head', 'cam_left_wrist', 'cam_right_wrist')"]
    for name, method in {**SpatialAPI.__dict__, **ManipulationAPI.__dict__, **RobotAPI.__dict__}.items():
        if name.startswith("_") or not inspect.isfunction(method):
            continue
        signature = str(inspect.signature(method)).replace("(self)", "()").replace("self, ", "").replace("__main__.", "")
        prefix = "async " if inspect.iscoroutinefunction(method) else ""
        lines.append(f"{prefix}robo.{name}{signature}\n{inspect.getdoc(method)}")
    for cls in (ArmState, State, Budget, Move, Gripper, Home, Region, Segment, ObjectSight, SeenPoint, Frame, Geometry, FrameImage, BoundFeature):
        names = ", ".join(f"{f.name}: {inspect.formatannotation(f.type)}" for f in fields(cls))
        lines.append(f"class {cls.__name__}({names})\n{inspect.getdoc(cls)}")
    lines.append("State.arm(name) selects an arm.\nprint/log emit text; await display(image) queues RGB/PNG for the next model turn.\nawait finish(status='failure'|'stuck', summary=str) stops the policy without physical action. Native success ends automatically.\nToolCallError reports a rejected operation. A cancelled cell loses Python state but retains physical effects.")
    return "\n\n".join(lines)


if __name__ == "__main__" and sys.argv[1:] == ["--describe"]:
    print(reference())
