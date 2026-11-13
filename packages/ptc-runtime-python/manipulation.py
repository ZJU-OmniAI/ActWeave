from copy import deepcopy
import math

import numpy as np
from spatial import SpatialAPI, Frame, Geometry, BoundFeature, _array, _unit, _world, _steps


def _wire(value):
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: _wire(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_wire(item) for item in value]
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(value):
            raise ValueError('motion parameters must be finite')
        return 0.0 if value == 0 else float(value)
    return value


def _motion_result(value):
    from robot_api import _state
    value['before'] = _state(value['before'])
    value['state'] = _state(value['state'])
    return value


def _matrix(frame):
    if isinstance(frame, Frame):
        matrix = np.eye(4)
        matrix[:3, :3] = frame.rotation
        matrix[:3, 3] = frame.origin
        return matrix
    return _array(frame, (4, 4))


def _rotation(axis, angle):
    x, y, z = _unit(axis)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + math.sin(angle)*cross + (1-math.cos(angle))*(cross@cross)


class ManipulationAPI(SpatialAPI):

    async def plan_feature_target(self, binding: BoundFeature, target: Geometry, rotations, *, axis_binding: BoundFeature | None = None, target_axis: Geometry | None = None, path: str = 'auto') -> dict:
        """Plan, never execute, caller-supplied TCP rotations that put a bound
        feature point at target. rotations is a nonempty list of 3x3 proper
        world TCP rotations. Optional axis_binding/target_axis report angular
        residuals without rejecting or selecting a candidate. All candidates
        retain original IK/path results and solver-FK feature residuals.
        Object bindings are rigid-motion assumptions, not current measurements.
        Astra chooses whether to execute a returned tcp_pose using move_pose.
        """
        if binding.feature.kind != 'point' or target.kind != 'point':
            raise ValueError('a bound point and target point are required')
        matrices = np.asarray(rotations, dtype=float)
        if matrices.ndim != 3 or matrices.shape[1:] != (3, 3) or not len(matrices) or not np.isfinite(matrices).all():
            raise ValueError('rotations requires a nonempty list of finite 3x3 rotations')
        if any(not np.allclose(r.T@r, np.eye(3), atol=1e-6, rtol=0) or not np.isclose(np.linalg.det(r), 1, atol=1e-6, rtol=0) for r in matrices):
            raise ValueError('candidate rotations must be proper orthonormal matrices')
        if (axis_binding is None) != (target_axis is None):
            raise ValueError('axis_binding and target_axis must be supplied together')
        if axis_binding is not None and (axis_binding.feature.kind != 'axis' or target_axis.kind != 'axis' or axis_binding.arm != binding.arm or axis_binding.tcp.step != binding.tcp.step):
            raise ValueError('axis binding must use the same arm and binding step; target_axis must be an axis')
        candidates = []
        for index, rotation in enumerate(matrices):
            pose = np.eye(4)
            pose[:3, :3] = rotation
            pose[:3, 3] = _world(target)-rotation@binding.local_value
            plan = await self.plan_pose(pose, binding.arm, path=path)
            fk = await self.kinematics(binding.arm, joints=plan['ik']['joints'])
            solved = np.asarray(fk['tcp_pose'])
            actual_point = solved[:3, 3]+solved[:3, :3]@binding.local_value
            candidate = {'index': index, 'tcp_pose': pose.tolist(), 'plan': plan,
                'solver_feature_position_error_m': float(np.linalg.norm(actual_point-_world(target))),
                'solver_tcp_pose': solved.tolist()}
            if axis_binding is not None:
                direction = _world(target_axis)
                candidate['requested_axis_error_deg'] = float(np.degrees(np.arccos(np.clip(np.dot(rotation@axis_binding.local_value, direction), -1, 1))))
                candidate['solver_axis_error_deg'] = float(np.degrees(np.arccos(np.clip(np.dot(solved[:3, :3]@axis_binding.local_value, direction), -1, 1))))
            candidates.append(candidate)
        return {'arm': binding.arm, 'binding': deepcopy(binding), 'target': deepcopy(target),
            'axis_binding': deepcopy(axis_binding), 'target_axis': deepcopy(target_axis),
            'candidates': candidates, 'executed': False}

    async def plan_pose(self, pose, arm: str = 'right', *, path: str = 'auto') -> dict:
        """Plan to a world 4x4 TCP pose or Frame without acting or collision vetoes. Return accepted IK or solver failure, search-domain residuals and route."""
        return await self._tools.plan_pose({'side': arm, 'pose': _wire(_matrix(pose)), 'path': path})

    async def move_pose(self, pose, arm: str = 'right', *, path: str = 'auto', gripper: int = 0, max_steps: int | None = None, max_joint_step_rad: float | None = None) -> dict:
        """Execute the supplied world 4x4 TCP pose or Frame without orientation substitution. Gripper: OPEN=-1, CLOSE=1, KEEP=0. Return actual state, solver results, steps, joint/pose residuals and waypoint progress. reached requires completed tracking and <=2 mm/1 deg; waypoint tracking uses min(0.001 rad, 10% initial joint error), floored at 1e-8 rad. No collision veto or stall stop; max_steps caps native actions; None uses the remaining native allowance. max_joint_step_rad optionally reduces the default 0.15-rad command increment; this is not force control. start_step/end_step delimit actual feedback. Returns a dict with before/state as State objects (result["state"].right.tcp_pos), tracking waypoint progress and motion_feedback for both arms; the latter distinguishes caller_max_steps/native_action_limit and reports commanded versus measured joints without inferring contact."""
        result = await self._tools.move_pose({'side': arm, 'pose': _wire(_matrix(pose)), 'path': path, 'gripper': gripper,
            **({'max_steps': int(max_steps)} if max_steps is not None else {}), **({'max_joint_step_rad': _wire(float(max_joint_step_rad))} if max_joint_step_rad is not None else {})})
        return _motion_result(result)

    async def move_relative(self, displacement, arm: str = 'right', *, frame: Frame | None = None, path: str = 'line', max_steps: int | None = None, max_joint_step_rad: float | None = None) -> dict:
        """Translate TCP by metres in world axes or frame axes while retaining orientation. Returns the same actual feedback and tracking thresholds as move_pose."""
        current = await self.tcp_frame(arm)
        delta = _array(displacement, (3,))
        target = _matrix(current)
        target[:3, 3] += frame.rotation@delta if frame is not None else delta
        result = await self.move_pose(target, arm, path=path, max_steps=max_steps, max_joint_step_rad=max_joint_step_rad)
        result['reference_steps'] = list(frame.steps if frame is not None else current.steps)
        return result

    async def contact_step(self, direction, distance_m: float, arm: str = 'right', *, frame: Frame | None = None, max_steps: int = 3, max_joint_step_rad: float | None = None) -> dict:
        """Execute one line displacement: unit direction in world/frame axes times signed distance_m. Return motion feedback, not a contact sensor reading."""
        return await self.move_relative(_unit(direction)*float(distance_m), arm, frame=frame, max_steps=max_steps, max_joint_step_rad=max_joint_step_rad)

    async def rotate_about(self, origin: Geometry, axis: Geometry, degrees: float, arm: str = 'right', *, step_degrees: float = 5.0, axial_advance_m: float = 0.0, max_steps: int | None = None, max_joint_step_rad: float | None = None) -> dict:
        """Rotate TCP pose about an explicit point/axis by degrees, with optional axial_advance_m. step_degrees sets trajectory sampling. Returns the same actual feedback and tracking thresholds as move_pose."""
        if origin.kind != 'point' or axis.kind != 'axis' or len(_steps(origin, axis)) != 1 or not math.isfinite(degrees) or not math.isfinite(step_degrees) or step_degrees <= 0:
            raise ValueError('one-step origin/axis and finite rotation parameters are required')
        current = await self.tcp_frame(arm)
        center, direction = _world(origin), _world(axis)
        count = max(1, math.ceil(abs(degrees)/step_degrees))
        poses = []
        for index in range(1, count+1):
            fraction = index/count
            rotation = _rotation(direction, math.radians(degrees*fraction))
            target = np.eye(4)
            target[:3, :3] = rotation@current.rotation
            target[:3, 3] = center+rotation@(current.origin-center)+direction*axial_advance_m*fraction
            poses.append(target)
        result = await self._tools.move_trajectory({'side': arm, 'poses': _wire(poses),
            **({'max_steps': int(max_steps)} if max_steps is not None else {}), **({'max_joint_step_rad': _wire(float(max_joint_step_rad))} if max_joint_step_rad is not None else {})})
        result['reference_steps'] = list(origin.steps)
        return _motion_result(result)

    async def kinematics(self, arm: str = 'right', *, joints=None) -> dict:
        """Read calibrated FK and TCP Jacobian at supplied/current joint radians. Jacobian rows are linear then angular; planning_joint_ranges_rad is the solver search domain."""
        return await self._tools.kinematics({'side': arm, **({'joints': _wire(_array(joints, (6,)))} if joints is not None else {})})

    async def feedback(self, before=None) -> dict:
        """Read public robot state and optional TCP/joint/opening deltas from before. Object contact, force and retention are not sensed."""
        state = await self.state()
        result = {'state': state, 'step': state.step, 'object_contact': None, 'force': None, 'retention': None}
        if before is not None:
            result['since_step'] = before.step
            result['arms'] = {arm: {'tcp_delta_m': (np.array(state.arm(arm).tcp_pos)-before.arm(arm).tcp_pos).tolist(),
                'joint_delta_rad': (np.array(state.arm(arm).joints)-before.arm(arm).joints).tolist(),
                'opening_delta': state.arm(arm).gripper-before.arm(arm).gripper} for arm in self.ARMS}
        return result


    async def record_feature(self, object_name: str, feature_name: str, value: Geometry) -> dict:
        """Record a named Geometry point or axis with its source and step, replacing this feature reference."""
        await self.record_object(object_name, evidence=value)
        self._objects[object_name].setdefault('features', {})[feature_name] = deepcopy(value)
        return {'object': object_name, 'feature': feature_name, 'value': value}

    async def object_features(self, object_name: str) -> dict:
        """Read named feature geometry and current_evidence flags at the current step."""
        step = (await self.state()).step
        return {name: {'geometry': deepcopy(value), 'current_evidence': value.steps == (step,)}
            for name, value in self._objects[object_name].get('features', {}).items()}


    def fit_opening(self, boundary: list[Geometry], *, shape: str = 'plane') -> dict:
        """Fit same-step boundary points. plane (default) retains the sample
        centroid frame/spans. Explicit circle fits a centre/radius in that
        plane, returning radial/plane residuals, angular coverage and solve
        condition. Circle shape and correspondences are caller assumptions;
        no interior, passage, contact or insertion success is established.
        """
        if shape not in ('plane', 'circle'):
            raise ValueError('shape must be plane or circle')
        if len(boundary) < 3 or any(point.kind != 'point' for point in boundary) or len(_steps(*boundary)) != 1:
            raise ValueError('at least three same-step boundary points are required')
        points = np.array([_world(point) for point in boundary])
        center = points.mean(axis=0)
        _, singular, axes = np.linalg.svd(points-center, full_matrices=False)
        if singular[1] < 1e-8:
            raise ValueError('collinear points do not define an opening plane')
        normal = np.cross(axes[0], axes[1])
        frame = Frame('selected_opening', boundary[0].steps[0], center, np.column_stack((axes[0], axes[1], normal)), 'selected_boundary_plane_fit', _steps(*boundary), tuple(boundary))
        local = (points-center)@frame.rotation
        result = {'frame': frame, 'span_m': np.ptp(local[:, :2], axis=0).tolist(),
            'radius_range_m': [float(np.linalg.norm(local[:, :2], axis=1).min()), float(np.linalg.norm(local[:, :2], axis=1).max())],
            'plane_residual_m': np.abs(local[:, 2]).tolist(), 'interior_and_passage': 'unobserved'}
        if shape == 'circle':
            xy = local[:, :2]
            scale = float(np.sqrt(np.mean(np.sum(xy*xy, axis=1))))
            normalized = xy/scale
            design = np.column_stack((2*normalized, np.ones(len(points))))
            solution, _, rank, singular = np.linalg.lstsq(design, np.sum(normalized*normalized, axis=1), rcond=None)
            radius_squared = solution[2]+np.dot(solution[:2], solution[:2])
            if rank < 3 or radius_squared <= 0:
                raise ValueError('boundary does not determine a nondegenerate circle')
            offset = solution[:2]*scale
            radius = float(np.sqrt(radius_squared)*scale)
            fitted_center = center+frame.rotation[:, :2]@offset
            distances = np.linalg.norm(xy-offset, axis=1)
            angles = np.sort(np.arctan2(xy[:, 1]-offset[1], xy[:, 0]-offset[0]))
            gap = float(np.diff(np.r_[angles, angles[0]+2*np.pi]).max())
            result.update(frame=Frame('selected_circle', frame.step, fitted_center, frame.rotation, 'selected_boundary_circle_fit', frame.steps, tuple(boundary)),
                shape='circle', sample_centroid=frame, radius_m=radius, radius_range_m=[float(distances.min()), float(distances.max())], radial_residual_m=(distances-radius).tolist(),
                angular_coverage_deg=float(np.degrees(2*np.pi-gap)), condition_number=float(singular[0]/singular[-1]))
        return result
