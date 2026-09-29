"""FrameResult ядра → ROS-сообщения (статус, объекты, vision_msgs, маркеры RViz).

Все выходы — в осях и frame_id входного облака: ядро отдаёт геометрию в осях
сообщения, пересчитывать ничего не нужно. Геометрия маркеров (коридор, рамки,
подписи, цвета) — в viz.py без ROS: её же пишет в MCAP bench/to_mcap.py.
"""

from __future__ import annotations

import math
from typing import Any

from geometry_msgs.msg import Point, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA, Header
from vision_msgs.msg import (
    BoundingBox3D,
    Detection3D,
    Detection3DArray,
    ObjectHypothesis,
    ObjectHypothesisWithPose,
)
from visualization_msgs.msg import Marker, MarkerArray

from metro_obstacle import viz
from metro_obstacle.viz import status_text
from metro_obstacle_msgs.msg import Obstacle, ObstacleArray, ObstacleStatus


def _f(v: float) -> float:
    return float(v)


def _quat_yaw(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))


def _point(p: Any) -> Point:
    return Point(x=_f(p[0]), y=_f(p[1]), z=_f(p[2]))


def _color(c: viz.RGBA) -> ColorRGBA:
    return ColorRGBA(r=c[0], g=c[1], b=c[2], a=c[3])


def status_msg(res: Any, header: Header, processing_ms: float) -> ObstacleStatus:
    return ObstacleStatus(
        header=header,
        obstacle=bool(res.obstacle),
        distance=_f(res.distance),
        confidence=_f(res.confidence),
        level=int(res.level),
        visible_range=_f(res.visible_range),
        processing_ms=_f(processing_ms),
        mode=str(res.mode),
        calibrated=bool(res.calibrated),
    )


def obstacles_msg(res: Any, header: Header) -> ObstacleArray:
    obs = viz.sorted_obstacles(res)
    return ObstacleArray(
        header=header,
        obstacles=[
            Obstacle(
                distance=_f(o.distance),
                lateral=_f(o.lateral),
                height=_f(o.height),
                length=_f(o.length),
                width=_f(o.width),
                points=int(o.points),
                confidence=_f(o.confidence),
                confirmed=bool(o.confirmed),
                position=_point(o.position),
                z_extent=_f(o.z_extent),
            )
            for o in obs
        ],
    )


def _box_pose(res: Any, o: Any) -> Pose:
    return Pose(position=_point(o.position), orientation=_quat_yaw(viz.box_yaw(res, o)))


def _box_size(o: Any) -> Vector3:
    x, y, z = viz.box_size(o)
    return Vector3(x=x, y=y, z=z)


def detections_msg(res: Any, header: Header) -> Detection3DArray:
    out = Detection3DArray(header=header)
    for k, o in enumerate(viz.sorted_obstacles(res)):
        pose = _box_pose(res, o)
        hyp = ObjectHypothesisWithPose(
            hypothesis=ObjectHypothesis(
                class_id="obstacle" if o.confirmed else "candidate", score=_f(o.confidence)
            )
        )
        hyp.pose.pose = pose
        out.detections.append(
            Detection3D(
                header=header,
                results=[hyp],
                bbox=BoundingBox3D(center=pose, size=_box_size(o)),
                id=str(k),
            )
        )
    return out


def _marker(header: Header, ns: str, mid: int, mtype: int, color: ColorRGBA) -> Marker:
    m = Marker(header=header, ns=ns, id=mid, type=mtype, action=Marker.ADD, color=color)
    m.pose.orientation.w = 1.0
    return m


def _corridor_markers(
    res: Any, header: Header, gauge_height: float, color: ColorRGBA
) -> list[Marker]:
    lines = viz.corridor_lines(res, gauge_height)
    if lines is None:
        return []
    out = []
    for k, edge in enumerate(lines.edges):
        m = _marker(header, "corridor", k, Marker.LINE_STRIP, color)
        m.scale.x = 0.06
        m.points = [_point(p) for p in edge]
        out.append(m)
    frames = _marker(header, "corridor", 4, Marker.LINE_LIST, color)
    frames.scale.x = 0.03
    frames.points = [_point(p) for p in lines.sections.reshape(-1, 3)]
    out.append(frames)
    return out


def markers_msg(res: Any, header: Header, gauge_height: float) -> MarkerArray:
    clear = Marker(header=header, action=Marker.DELETEALL)
    markers = [clear]
    markers += _corridor_markers(res, header, gauge_height, _color(viz.level_color(res)))

    for k, o in enumerate(viz.sorted_obstacles(res)):
        color = _color(viz.obstacle_color(o))
        box = _marker(header, "obstacles", k, Marker.CUBE, color)
        box.pose = _box_pose(res, o)
        box.scale = _box_size(o)
        markers.append(box)

        label = _marker(header, "obstacle_labels", k, Marker.TEXT_VIEW_FACING, color)
        label.pose.position = _point(viz.label_position(o, lift=0.6))
        label.scale.z = 0.6
        label.text = f"{o.distance:.1f} m"
        markers.append(label)

    title = _marker(
        header, "status", 0, Marker.TEXT_VIEW_FACING, _color(viz.level_color(res, viz.WHITE))
    )
    title.pose.position.z = 2.5
    title.scale.z = 1.0
    title.text = status_text(res)
    markers.append(title)
    return MarkerArray(markers=markers)
