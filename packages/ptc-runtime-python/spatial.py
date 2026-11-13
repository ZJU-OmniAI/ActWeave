from copy import deepcopy
from dataclasses import dataclass, field
import math

import numpy as np
from PIL import Image, ImageDraw


def _array(value, shape):
    result = np.asarray(value, dtype=float)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f"expected finite array with shape {shape}")
    return result.copy()


def _step(value):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError("step must be a nonnegative observation index")
    return int(value)


def _unit(value):
    result = _array(value, (3,))
    length = np.linalg.norm(result)
    if length < 1e-12:
        raise ValueError("zero direction has no axis")
    return result / length


@dataclass(frozen=True)
class Frame:
    """Local-to-world rigid frame: origin in metres, rotation columns are local axes in world, with source and observation steps."""
    name: str
    step: int
    origin: np.ndarray
    rotation: np.ndarray
    source: str
    steps: tuple[int, ...]
    evidence: tuple
    unit: str = "m"


@dataclass(frozen=True)
class Geometry:
    """Point in metres or directed unit axis in a Frame. steps/source/evidence retain its observation or assumption. sensitivity, when present, is a world-coordinate pixel Jacobian with fixed calibration/model, not physical accuracy or confidence."""
    kind: str
    value: np.ndarray
    frame: Frame
    source: str
    steps: tuple[int, ...]
    evidence: tuple
    unit: str = field(init=False)
    sensitivity: dict | None = None

    def __post_init__(self):
        object.__setattr__(self, 'unit', 'm' if self.kind == 'point' else 'unit_vector')


@dataclass(frozen=True)
class FrameImage:
    """RGB array and source camera/step/crop mappings. sources preserves original pixel coordinates through crop or montage."""
    image: np.ndarray = field(repr=False)
    sources: tuple[dict, ...]


@dataclass(frozen=True)
class BoundFeature:
    """A caller-assumed rigid feature-to-TCP relation at one step, not a retention measurement."""
    feature: Geometry
    tcp: Frame
    arm: str
    local_value: np.ndarray
    assumption: str = 'rigid_to_tcp'


def _world(value):
    return value.frame.rotation @ value.value + (value.frame.origin if value.kind == 'point' else 0)


def _steps(*values):
    return tuple(sorted({step for value in values for step in value.steps}))


class SpatialAPI:
    def world_frame(self, step: int) -> Frame:
        """World axes at an explicit observation step, with metre coordinates."""
        step = _step(step)
        return Frame('world', step, np.zeros(3), np.eye(3), 'world_coordinates', (step,), ())

    def local_frame(self, name: str, origin: Geometry, x_axis: Geometry, y_axis: Geometry) -> Frame:
        """Define a local/object frame from one point and two directed axes at
        the same step. x and y must be orthogonal; z=x cross y. This definition
        retains the supplied measurements/assignments, not a full object-pose
        estimate. No frame is inferred from an object name.
        """
        if not name or origin.kind != 'point' or x_axis.kind != 'axis' or y_axis.kind != 'axis':
            raise ValueError('name, origin point and x/y axes are required')
        steps = _steps(origin, x_axis, y_axis)
        if len(steps) != 1:
            raise ValueError('a local frame requires inputs from the same observation step')
        x, y = _world(x_axis), _world(y_axis)
        if abs(np.dot(x, y)) > 1e-6:
            raise ValueError('local x/y axes must be orthogonal')
        return Frame(name, steps[0], _world(origin), np.column_stack((x, y, np.cross(x, y))), 'defined_from_geometry', steps, (origin, x_axis, y_axis))

    async def camera_frame(self, camera: str = 'cam_head', *, step: int = -1) -> Frame:
        """Public calibrated camera frame: +x right, +y up, forward -z.
        The returned step is resolved now, including for a wrist camera.
        """
        meta = await self.camera_meta(camera, step=step)
        matrix = np.asarray(meta['extrinsic_matrix'])
        return Frame(camera, meta['step'], matrix[:3, 3], matrix[:3, :3], 'public_camera_calibration', (meta['step'],), ({'camera': camera, 'step': meta['step']},))

    async def tcp_frame(self, arm: str = 'right', *, step: int = -1) -> Frame:
        """Public TCP frame: origin at finger-pad midpoint, axes parallel to
        link6 (+x approach, +y jaw separation, +z completing the frame).
        Metres, quaternion wxyz; no held-object pose is inferred.
        """
        state = await self.state(step)
        value = state.arm(arm)
        w, x, y, z = np.asarray(value.eef_quat) / np.linalg.norm(value.eef_quat)
        rotation = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)], [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)], [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        return Frame('tcp_' + arm, state.step, np.asarray(value.tcp_pos), rotation, 'public_robot_state', (state.step,), ())

    def point(self, xyz, *, frame: Frame) -> Geometry:
        """Explicitly assign a point in frame, in metres. source is always
        astra_assignment; a coordinate typed by Astra is not a measurement.
        Use triangulate, intersect_plane or sight_point for measured/derived
        values. Assigning [0,0,0] in a public TCP frame denotes its origin.
        """
        return Geometry('point', _array(xyz, (3,)), frame, 'astra_assignment', frame.steps, ())

    def axis(self, direction, *, frame: Frame) -> Geometry:
        """Explicitly assign a directed local axis, normalized to unit length.
        Zero axes raise ValueError. source remains astra_assignment.
        """
        return Geometry('axis', _unit(direction), frame, 'astra_assignment', frame.steps, ())

    def axis_between(self, a: Geometry, b: Geometry) -> Geometry:
        """Directed world axis from a to b at one observation step. Retains
        the two points' sources and evidence; coincident points fail.
        """
        if a.kind != 'point' or b.kind != 'point' or len(_steps(a, b)) != 1:
            raise ValueError('axis_between requires two points at one step')
        return Geometry('axis', _unit(_world(b)-_world(a)), self.world_frame(a.steps[0]), 'point_pair_direction', a.steps, (a, b))

    def sight_point(self, sight) -> Geometry:
        """Use the CAP fitted centre as a world Geometry point, retaining its source and step. Missing geometry raises ValueError."""
        if sight.center_xyz is None:
            raise ValueError('ObjectSight has no usable 3D centre: ' + str(sight.note))
        return Geometry('point', _array(sight.center_xyz, (3,)), self.world_frame(sight.step), 'cap_fitted_center', (sight.step,), (deepcopy(sight),))

    def transform(self, value: Geometry, frame: Frame) -> Geometry:
        """Express a point/axis in another dated frame, preserving all input
        steps and sources. Different steps remain explicit; this operation
        does not move an old object estimate forward in time.
        """
        local = frame.rotation.T @ (_world(value) - (frame.origin if value.kind == 'point' else 0))
        return Geometry(value.kind, local, frame, value.source, _steps(value, frame), (value,), sensitivity=deepcopy(value.sensitivity))

    def bind_feature_to_tcp(self, feature: Geometry, tcp: Frame) -> BoundFeature:
        """Bind a point/axis to an explicitly supplied same-step public TCP frame.
        This records your rigid-motion assumption, not a detected grasp. Regrasp,
        release and handover do not automatically replace or transfer a binding.
        """
        if feature.kind not in ('point', 'axis') or tcp.source != 'public_robot_state' or tcp.name not in ('tcp_left', 'tcp_right'):
            raise ValueError('a point/axis and a public TCP frame are required')
        if feature.steps != (tcp.step,) or tcp.steps != (tcp.step,):
            raise ValueError('binding inputs must refer to the same observation step')
        local = tcp.rotation.T @ (_world(feature) - (tcp.origin if feature.kind == 'point' else 0))
        return BoundFeature(deepcopy(feature), deepcopy(tcp), tcp.name.removeprefix('tcp_'), local)

    async def predict_bound_feature(self, binding: BoundFeature, *, step: int = -1) -> Geometry:
        """Predict a bound feature from the named arm's public pose at step.
        source=rigid_prediction; steps/evidence retain binding and prediction
        times. No object is reobserved and no automatic tracking occurs.
        """
        tcp = await self.tcp_frame(binding.arm, step=step)
        value = tcp.rotation @ binding.local_value + (tcp.origin if binding.feature.kind == 'point' else 0)
        return Geometry(binding.feature.kind, value, self.world_frame(tcp.step), 'rigid_prediction',
            _steps(binding.feature, tcp), (deepcopy(binding), tcp))

    async def compare_bound_feature(self, binding: BoundFeature, observed: Geometry) -> dict:
        """Compare rigid prediction with supplied same-kind, single-step geometry.
        Returns observed-minus-predicted metres or directed angular difference,
        with sources. The difference is not a slip diagnosis or action gate.
        """
        if observed.kind != binding.feature.kind or len(observed.steps) != 1:
            raise ValueError('same-kind geometry from one observation step is required')
        predicted = await self.predict_bound_feature(binding, step=observed.steps[0])
        a, b = _world(predicted), _world(observed)
        result = {'prediction': predicted, 'observation': deepcopy(observed), 'step': observed.steps[0],
            'binding_step': binding.tcp.step, 'retention': 'unobserved'}
        if observed.kind == 'point':
            result.update(displacement_m=(b-a).tolist(), distance_m=float(np.linalg.norm(b-a)))
        else:
            result['angle_deg'] = float(np.degrees(np.arccos(np.clip(np.dot(a, b), -1, 1))))
        return result

    async def pixel_ray(self, row: float, col: float, camera: str = 'cam_head', *, step: int = -1) -> dict:
        """World unit ray from a specified original RGB pixel [row,col].
        Returns origin/direction Geometry, direction_per_pixel (world rows,
        [row,col] derivative columns) and original frame reference. The
        pixel is chosen by Astra; calibration alone provides no depth.
        """
        pixel = _array([row, col], (2,))
        meta = await self.camera_meta(camera, step=step)
        h, w = meta['shape'][:2]
        if not (0 <= row < h and 0 <= col < w):
            raise ValueError('pixel is outside the original image')
        k, m = np.asarray(meta['intrinsic_matrix']), np.asarray(meta['extrinsic_matrix'])
        frame = self.world_frame(meta['step'])
        evidence = ({'camera': camera, 'step': meta['step'], 'pixel': pixel.tolist()},)
        direction = m[:3, :3] @ np.array([(col-k[0, 2])/k[0, 0], -(row-k[1, 2])/k[1, 1], -1])
        unit = _unit(direction)
        derivatives = (np.eye(3)-np.outer(unit, unit)) @ m[:3, :3] @ np.array([[0, 1/k[0, 0]], [-1/k[1, 1], 0], [0, 0]]) / np.linalg.norm(direction)
        return {'origin': Geometry('point', m[:3, 3], frame, 'public_camera_calibration', frame.steps, evidence), 'direction': Geometry('axis', unit, frame, 'calibrated_pixel_ray', frame.steps, evidence), 'frame': evidence[0], 'direction_per_pixel': derivatives}

    def intersect_plane(self, ray: dict, origin: Geometry, normal: Geometry) -> Geometry:
        """Intersect a forward ray with the explicitly supplied plane. The
        plane is an assumption/measurement supplied by Astra, not measured
        depth. Parallel rays, behind-camera intersections or mixed times fail.
        Result sensitivity reports the world Jacobian per pixel under the
        fixed-plane assumption, not absolute accuracy.
        """
        if origin.kind != 'point' or normal.kind != 'axis':
            raise ValueError('plane requires an origin point and normal axis')
        steps = _steps(ray['origin'], ray['direction'], origin, normal)
        if len(steps) != 1:
            raise ValueError('ray and plane must refer to one observation step')
        o, d, n = _world(ray['origin']), _world(ray['direction']), _world(normal)
        divisor = np.dot(d, n)
        if abs(divisor) < 1e-10:
            raise ValueError('ray is parallel to plane')
        distance = np.dot(_world(origin)-o, n)/divisor
        if distance <= 0:
            raise ValueError('plane intersection is behind camera')
        jacobian = distance * (np.eye(3)-np.outer(d, n)/divisor) @ ray['direction_per_pixel']
        sensitivity = {'world_jacobian_m_per_px': jacobian.tolist(), 'pixel_order': ['row', 'col'],
            'worst_direction_m_per_px': float(np.linalg.norm(jacobian, 2)), 'assumption': 'fixed plane and calibration; local sensitivity, not error probability'}
        return Geometry('point', o+distance*d, self.world_frame(steps[0]), 'specified_plane_intersection', steps, (deepcopy(ray), origin, normal), sensitivity=sensitivity)

    def triangulate(self, rays: list[dict]) -> dict:
        """Least-squares point from >=2 chosen pixel rays at one observation
        step. Astra establishes correspondence; no object matching runs.
        Degenerate/parallel rays or behind-camera solutions fail. Returns point,
        per-ray residual_m, condition_number and a world pixel Jacobian plus
        worst-direction sensitivity. Calibration/correspondence are held fixed;
        these are not error probabilities or correspondence validation.
        """
        if len(rays) < 2:
            raise ValueError('triangulation requires at least two rays')
        steps = _steps(*(r[key] for r in rays for key in ('origin', 'direction')))
        if len(steps) != 1:
            raise ValueError('triangulation requires simultaneous observations')
        origins = np.array([_world(r['origin']) for r in rays])
        directions = np.array([_world(r['direction']) for r in rays])
        projectors = np.eye(3)[None, :, :] - directions[:, :, None]*directions[:, None, :]
        matrix = projectors.sum(axis=0)
        condition = float(np.linalg.cond(matrix))
        if not math.isfinite(condition) or condition > 1e10:
            raise ValueError('degenerate ray geometry')
        point = np.linalg.solve(matrix, np.einsum('nij,nj->i', projectors, origins))
        if np.any(np.einsum('ni,ni->n', point-origins, directions) <= 0):
            raise ValueError('triangulation lies behind a camera')
        residual = np.linalg.norm(np.einsum('nij,nj->ni', projectors, point-origins), axis=1)
        columns = []
        for ray, origin, direction in zip(rays, origins, directions):
            for derivative in np.asarray(ray['direction_per_pixel']).T:
                change = -np.outer(derivative, direction)-np.outer(direction, derivative)
                columns.append(np.linalg.solve(matrix, change @ (origin-point)))
        jacobian = np.column_stack(columns)
        sensitivity = {'world_jacobian_m_per_px': jacobian.tolist(), 'pixel_order': [[i, axis] for i in range(len(rays)) for axis in ('row', 'col')],
            'worst_direction_m_per_px': float(np.linalg.norm(jacobian, 2)), 'assumption': 'fixed correspondence and calibration; joint pixel-perturbation L2 norm of one; not error probability'}
        return {'point': Geometry('point', point, self.world_frame(steps[0]), 'pixel_ray_triangulation', steps, tuple(deepcopy(rays)), sensitivity=sensitivity), 'residual_m': residual.tolist(), 'condition_number': condition, 'sensitivity': sensitivity}

    async def project(self, point: Geometry, camera: str = 'cam_head', *, step: int = -1) -> dict:
        """Project a point onto an original RGB frame; return floating [row,col],
        optical depth_m, in_image and all input steps. Behind-camera points
        raise ValueError. same_step=false explicitly marks a historical overlay.
        """
        if point.kind != 'point':
            raise ValueError('projection requires a point')
        meta = await self.camera_meta(camera, step=step)
        m, k = np.asarray(meta['extrinsic_matrix']), np.asarray(meta['intrinsic_matrix'])
        x, y, z = m[:3, :3].T @ (_world(point)-m[:3, 3])
        if z >= -1e-12:
            raise ValueError('point is behind camera')
        row, col = k[1, 2]+k[1, 1]*y/z, k[0, 2]-k[0, 0]*x/z
        steps = tuple(sorted(set(point.steps+(meta['step'],))))
        return {'pixel': [float(row), float(col)], 'depth_m': float(-z), 'in_image': bool(0 <= row < meta['shape'][0] and 0 <= col < meta['shape'][1]), 'camera': camera, 'step': meta['step'], 'steps': steps, 'same_step': len(steps) == 1, 'source': point.source}

    async def match_texture(self, row: float, col: float, camera: str, target_camera: str, *, distance_range_m, step: int = -1, samples: int = 256, patch_radius: int = 4, top_k: int = 5, separation_px: float = 3.0) -> dict:
        """Return candidate corresponding [row,col] pixels in another same-step
        camera, using normalized mean-RGB patch correlation along a calibrated
        epipolar search. distance_range_m=[near,far] is your explicit distance
        interval along the source unit ray, not optical z. samples controls a
        uniform distance grid; patch_radius sets an unwarped square patch.
        Returns ranked candidates, correlation scores, pixel/ray-distance
        provenance and source contrast. No candidate is certified or used for
        automatic triangulation. Low/repeated texture, occlusion and viewpoint
        changes can make correlation misleading. No image is resized or shown.
        """
        interval = _array(distance_range_m, (2,))
        if camera == target_camera or interval[0] <= 0 or interval[1] <= interval[0]:
            raise ValueError('distinct cameras and 0 < near < far are required')
        if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (samples, patch_radius, top_k)) or samples < 2 or patch_radius < 1 or top_k < 1:
            raise ValueError('samples >= 2, patch_radius >= 1 and top_k >= 1 must be integers')
        if not math.isfinite(separation_px) or separation_px <= 0:
            raise ValueError('separation_px must be finite and positive')
        ray = await self.pixel_ray(row, col, camera, step=step)
        at = ray['frame']['step']
        first = await self.frame(camera, step=at)
        second = await self.frame(target_camera, step=at)
        meta = await self.camera_meta(target_camera, step=at)
        h, w = first.image.shape[:2]
        if not (patch_radius <= row <= h-1-patch_radius and patch_radius <= col <= w-1-patch_radius):
            raise ValueError('source patch must lie entirely inside the image')
        offsets = np.stack(np.meshgrid(np.arange(-patch_radius, patch_radius+1), np.arange(-patch_radius, patch_radius+1), indexing='ij'), axis=-1).reshape(-1, 2)

        def patches(image, centers):
            gray = image.astype(float).mean(axis=2)
            pixels = centers[:, None, :] + offsets
            low = np.floor(pixels).astype(int)
            high = np.minimum(low+1, np.array(gray.shape)-1)
            fraction = pixels-low
            a, b = fraction[:, :, 0], fraction[:, :, 1]
            return (1-a)*((1-b)*gray[low[:, :, 0], low[:, :, 1]]+b*gray[low[:, :, 0], high[:, :, 1]]) + a*((1-b)*gray[high[:, :, 0], low[:, :, 1]]+b*gray[high[:, :, 0], high[:, :, 1]])

        template = patches(first.image, np.array([[row, col]]))[0]
        contrast = float(template.std())
        result = {'source': 'rgb_patch_correlation_candidates', 'step': at, 'camera': camera, 'pixel': [row, col],
            'target_camera': target_camera, 'distance_range_m': interval.tolist(), 'samples': samples,
            'patch_radius': patch_radius, 'source_contrast_std': contrast, 'candidates': [],
            'correspondence': 'unverified', 'score_gap': None}
        template -= template.mean()
        energy = np.linalg.norm(template)
        if energy == 0:
            return {**result, 'reason': 'constant_source_patch'}
        distances = np.linspace(*interval, samples)
        points = _world(ray['origin']) + distances[:, None]*_world(ray['direction'])
        matrix, k = np.asarray(meta['extrinsic_matrix']), np.asarray(meta['intrinsic_matrix'])
        local = (points-matrix[:3, 3]) @ matrix[:3, :3]
        front = np.flatnonzero(local[:, 2] < -1e-12)
        local = local[front]
        pixels = np.column_stack((k[1, 2]+k[1, 1]*local[:, 1]/local[:, 2], k[0, 2]-k[0, 0]*local[:, 0]/local[:, 2]))
        h, w = second.image.shape[:2]
        inside = (pixels[:, 0] >= patch_radius) & (pixels[:, 0] <= h-1-patch_radius) & (pixels[:, 1] >= patch_radius) & (pixels[:, 1] <= w-1-patch_radius)
        pixels, indices = pixels[inside], front[inside]
        if not len(indices):
            return {**result, 'reason': 'no_in_view_search_patches'}
        values = patches(second.image, pixels)
        values -= values.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(values, axis=1)
        usable = norms > 0
        pixels, indices, values, norms = pixels[usable], indices[usable], values[usable], norms[usable]
        scores = np.clip(values @ template / (norms*energy), -1, 1)
        selected = []
        for index in np.argsort(-scores):
            if any(np.linalg.norm(pixels[index]-pixels[other]) < separation_px for other in selected):
                continue
            selected.append(index)
            if len(selected) == top_k:
                break
        result['candidates'] = [{'pixel': pixels[i].tolist(), 'correlation': float(scores[i]), 'source_ray_distance_m': float(distances[indices[i]])} for i in selected]
        result['score_gap'] = float(scores[selected[0]]-scores[selected[1]]) if len(selected) > 1 else None
        result['reason'] = 'candidates_returned' if selected else 'constant_target_patches'
        return result

    def relation(self, a: Geometry, b: Geometry, *, frame: Frame) -> dict:
        """Points: b-a displacement in frame and Euclidean distance in metres.
        Axes: directed angle in [0,180] degrees. Return all observation steps
        and same_step; mixed times describe a historical comparison, never a
        current relation. Source geometries remain attached; no contact/force,
        retention, complete pose or task success is inferred.
        """
        if a.kind != b.kind:
            raise ValueError('relation requires two points or two axes')
        steps = _steps(a, b, frame)
        result = {'frame': frame.name, 'frame_step': frame.step, 'steps': steps, 'same_step': len(steps) == 1, 'inputs': (a, b)}
        if a.kind == 'point':
            displacement = frame.rotation.T @ (_world(b)-_world(a))
            result.update(displacement=displacement.tolist(), distance=float(np.linalg.norm(displacement)), unit='m')
        else:
            result.update(angle=float(np.degrees(np.arccos(np.clip(np.dot(_world(a), _world(b)), -1, 1)))), unit='deg')
        return result

    async def record_object(self, name: str, *, description: str | None = None, role: str | None = None, evidence=None, judgment: str | None = None, frames: list[dict] | None = None, query: dict | None = None) -> dict:
        """Create/revise Astra's named instance reference. description is the
        reusable locate query; role and judgment are Astra assertions. Evidence
        accepts ObjectSight, Segment or Geometry, with its original step and
        uncertainty. frames explicitly names original camera/step pairs.
        query can retain locate camera/text_prompt/point/box/view_prompts and
        region selection. Every correction is appended; no completed flag is inferred. Records live in persistent Python and are lost on reset.
        """
        from robot_api import ObjectSight, Segment
        if not isinstance(name, str) or not name.strip():
            raise ValueError('object name must be nonempty')
        if any(value is not None and not isinstance(value, str) for value in (description, role, judgment)):
            raise ValueError('description, role and judgment must be strings')
        if evidence is not None and not isinstance(evidence, (ObjectSight, Segment, Geometry)):
            raise ValueError('evidence must be ObjectSight, Segment or Geometry')
        refs = deepcopy(frames or [])
        for ref in refs:
            if ref['camera'] not in self.CAMERAS:
                raise ValueError('unknown camera')
            _step(ref['step'])
        if isinstance(evidence, (ObjectSight, Segment)):
            refs.append({'camera': evidence.camera, 'step': evidence.step})
            if isinstance(evidence, ObjectSight):
                refs.extend({'camera': view['camera'], 'step': evidence.step} for view in evidence.view_support if view.get('selected') and view['camera'] != evidence.camera)
        state = await self.state()
        record = self._objects.setdefault(name, {'name': name, 'description': None, 'role': None, 'history': []})
        if description is not None:
            record['description'] = description
        if role is not None:
            record['role'] = role
        if query is not None:
            record['query'] = deepcopy(query)
        record['history'].append({'recorded_step': state.step, 'description': record['description'], 'role': record['role'], 'evidence': deepcopy(evidence), 'judgment': judgment, 'frames': refs, 'query': deepcopy(record.get('query', {}))})
        return await self.object_record(name)

    async def object_record(self, name: str) -> dict:
        """Read a named reference and its correction history. evidence_steps
        and current_evidence distinguish current measurement from history at
        the sampled observation step. A judgment always remains Astra's claim.
        No older evidence is relabelled by a later name/role/judgment edit.
        """
        record = deepcopy(self._objects[name])
        step = (await self.state()).step
        for entry in record['history']:
            value = entry['evidence']
            steps = value.steps if isinstance(value, Geometry) else (value.step,) if value is not None else ()
            entry.update(evidence_steps=steps, current_evidence=bool(steps) and all(s == step for s in steps))
        record['sampled_step'] = step
        return record

    async def measure_object(self, name: str, *, camera: str | None = None, min_score: float | None = None):
        """Run CAP locate using the named object description or saved query, then append the result to its evidence history."""
        record = self._objects[name]
        query = deepcopy(record.get('query', {}))
        if not any(key in query for key in ('text_prompt', 'point', 'box')):
            if not record['description']:
                raise ValueError('set an object description or explicit query before measuring')
            query['text_prompt'] = record['description']
        if camera is not None:
            query['camera'] = camera
        if min_score is not None:
            query['min_score'] = min_score
        sight = await self.locate(**query)
        await self.record_object(name, evidence=sight)
        return sight

    async def objects(self) -> dict:
        """Return copies of all explicitly named object records at the current
        step. Does not measure, associate identities, infer holding or update poses.
        """
        return {name: await self.object_record(name) for name in self._objects}

    async def frame(self, camera: str = 'cam_head', *, step: int = -1) -> FrameImage:
        """Capture full original RGB now, with a fixed camera/observation step.
        No action runs. display(frame.image) sends selected pixels to Astra.
        """
        from robot_api import _rgb
        value = await self._tools.image({'camera': camera, 'step': int(step)})
        pixels = _rgb(value['png_base64'])
        return FrameImage(pixels, ({'camera': camera, 'step': value['step'], 'box': [0, 0, pixels.shape[1], pixels.shape[0]], 'offset': [0, 0]},))

    def crop(self, frame: FrameImage, box) -> FrameImage:
        """Crop one original/cropped frame, box=[left,top,right,bottom] in
        input-image pixels. Preserve the original camera/step and source box.
        No resize, padding, calibration substitution or original overwrite.
        """
        if len(frame.sources) != 1 or frame.sources[0]['offset'] != [0, 0]:
            raise ValueError('crop individual frames before arranging a montage')
        values = _array(box, (4,))
        if not np.equal(values, np.floor(values)).all():
            raise ValueError('crop bounds must be integers')
        left, top, right, bottom = values.astype(int)
        h, w = frame.image.shape[:2]
        if not (0 <= left < right <= w and 0 <= top < bottom <= h):
            raise ValueError('crop box is outside image')
        source = deepcopy(frame.sources[0])
        x, y = source['box'][:2]
        source['box'] = [int(x+left), int(y+top), int(x+right), int(y+bottom)]
        return FrameImage(frame.image[top:bottom, left:right].copy(), (source,))

    def montage(self, frames: list[FrameImage], *, columns: int = 3) -> FrameImage:
        """Arrange full frames/crops at original pixel scale, labelled by
        camera, step and original crop box. sources retains offsets to recover
        pixel provenance. Display only when useful; complete originals remain.
        """
        if not frames or not isinstance(columns, int) or columns < 1 or any(len(f.sources) != 1 or f.sources[0]['offset'] != [0, 0] for f in frames):
            raise ValueError('provide individual frames and positive columns')
        columns = min(columns, len(frames))
        width = max(f.image.shape[1] for f in frames)
        height = max(f.image.shape[0] for f in frames)+32
        canvas = Image.new('RGB', (width*columns, height*math.ceil(len(frames)/columns)))
        draw, sources = ImageDraw.Draw(canvas), []
        for i, frame in enumerate(frames):
            x, y = i % columns*width, i // columns*height
            source = deepcopy(frame.sources[0])
            source['offset'] = [x, y+32]
            draw.text((x+2, y+2), f"{source['camera']} step {source['step']}\nbox {source['box']}", fill='white')
            canvas.paste(Image.fromarray(frame.image), (x, y+32))
            sources.append(source)
        return FrameImage(np.array(canvas), tuple(sources))

    async def overlay(self, frame: FrameImage, *, points: dict | None = None, axes: dict | None = None, masks: dict | None = None) -> FrameImage:
        """Annotate a single original/cropped frame. points={name: Geometry};
        axes={name: (origin_point, axis, length_m)}; masks={name: (Segment,
        region_index)}. Geometry and masks must match the original frame step.
        Projection uses original calibration before crop offset. Names label
        Astra's references, not verified identities; original pixels survive.
        """
        if len(frame.sources) != 1 or frame.sources[0]['offset'] != [0, 0]:
            raise ValueError('overlay individual frames before montage')
        source = frame.sources[0]
        camera, step = source['camera'], source['step']
        left, top, right, bottom = source['box']
        pixels = frame.image.copy()
        labels = []
        for name, (segment, index) in (masks or {}).items():
            if segment.camera != camera or segment.step != step:
                raise ValueError('mask and original frame must match')
            region = segment.regions[index]
            mask = region.mask[top:bottom, left:right]
            pixels[mask] = (pixels[mask]*0.55+np.array([0, 220, 180])*0.45).astype(np.uint8)
            labels.append((str(name), (region.centroid[1]-left, region.centroid[0]-top)))
        canvas = Image.fromarray(pixels)
        draw = ImageDraw.Draw(canvas)
        async def pixel(value):
            if value.steps != (step,):
                raise ValueError('geometry and original frame must share one step; use project for explicit historical comparison')
            result = await self.project(value, camera, step=step)
            row, col = result['pixel']
            return col-left, row-top
        for name, value in (points or {}).items():
            x, y = await pixel(value)
            draw.ellipse((x-3, y-3, x+3, y+3), outline='yellow', width=2)
            labels.append((str(name), (x+4, y+4)))
        for name, (origin, axis, length) in (axes or {}).items():
            if origin.kind != 'point' or axis.kind != 'axis' or not math.isfinite(length) or length <= 0:
                raise ValueError('axis overlay requires a point, axis and positive metre length')
            if axis.steps != (step,):
                raise ValueError('axis and original frame steps differ')
            endpoint = Geometry('point', _world(origin)+length*_world(axis), self.world_frame(step), 'axis_endpoint', _steps(origin, axis), (origin, axis))
            start, end = await pixel(origin), await pixel(endpoint)
            draw.line((start, end), fill='red', width=2)
            draw.ellipse((end[0]-3, end[1]-3, end[0]+3, end[1]+3), fill='red')
            labels.append((str(name), end))
        for name, at in labels:
            draw.text(at, name, fill='yellow', stroke_width=1, stroke_fill='black')
        draw.text((2, 2), f"{camera} step {step}", fill='white', stroke_width=1, stroke_fill='black')
        return FrameImage(np.array(canvas), deepcopy(frame.sources))
