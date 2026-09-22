import numpy as np
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Any

def parse_cyberfungi(
    xml_path: str,
    dist_lambda: float = 0.02,
    angle_mu: float = 1.0,
    merge_tol: float = 1e-6
) -> Tuple[
        Dict[int, np.ndarray], 
        List[Tuple[int, int, float, float, int]], 
        Dict[int, Dict[str, Any]], 
        Dict[int, List[int]]
    ]:
    """Parse CyberFungi XML simulation output into network data structures.
    
    Args:
        xml_path: Path to the .mycelia.xml file
        dist_lambda: Distance decay parameter for edge weights
        angle_mu: Angle penalty multiplier for edge weights
        merge_tol: Tolerance for merging close nodes (coordinate quantization)
        
    Returns:
        Tuple containing:
            - node_coords: Dict mapping node ID -> (x, y, z) coordinates
            - weighted_edges: List of (u, v, weight, distance, section_id) tuples
            - sections: Dict mapping section ID -> {parent, pts, diameter, age}
            - children: Dict mapping parent section ID -> list of child section IDs
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()
    mycelia = root.find(".//mycelia")
    assert mycelia
    secs = mycelia.findall("sec")

    # collect section geometry
    sections: Dict[int, Dict[str, Any]] = {}
    children: Dict[int, List[int]] = defaultdict(list)
    for sec in secs:
        sid = int(sec.attrib["i"])
        parent = int(sec.attrib.get("b", "0"))
        pts = parse_segs(sec)
        sections[sid] = {
            "parent": parent,
            "pts": pts,
            "diameter": float(sec.attrib.get("d", "0")),
            "age": float(sec.attrib.get("s", "0")),
        }
        children[parent].append(sid)

    node_ids: Dict[Tuple[int, int, int], int] = {}
    node_coords: Dict[int, np.ndarray] = {}

    # point (node_coords) are just (x, y, z) float tuples
    def get_node(point: np.ndarray) -> int:
        """point is (x, y, z) tuple"""
        
        # key by rounded coordinates (merges close points)
        key = tuple(np.round(point / merge_tol).astype(int))
        if key not in node_ids:
            nid = len(node_ids)
            node_ids[key] = nid
            node_coords[nid] = point.astype(float)

        return node_ids[key]

    # calculate weighted edges [(pt1, pt2, weight, distance, sid)]
    weighted_edges: List[Tuple[int, int, float, float, int]] = []
    for sid, sec in sections.items():
        pts = sec["pts"]
        if len(pts) > 2:
            for i in range(len(pts) - 1):
                a = pts[i]
                b = pts[i + 1]
                u = get_node(a)
                v = get_node(b)

                dist = float(np.linalg.norm(b - a))
                w = np.exp(-dist_lambda * dist)
                if 0 < i < len(pts) - 1:
                    theta = calc_angle(pts[i - 1], pts[i], pts[i + 1])
                    w *= np.exp(-angle_mu * theta)

                weighted_edges.append((u, v, w, dist, sid))

    return node_coords, weighted_edges, sections, children

def parse_segs(sec: ET.Element) -> Optional[List[np.ndarray]]:
    """Parse segment points from a section element.
    
    Args:
        sec: XML element representing a section/segment
        
    Returns:
        List of numpy arrays, each containing (x, y, z) coordinates
    """
    segs = sec.findtext("segs")
    if segs and segs.strip() != "":
        pts = []
        for chunk in segs.split(","):
            point = np.array([float(x) for x in chunk.strip().split()], dtype=float)
            pts.append(point)
        return pts
    else:
        y = sec.attrib.get("y")
        if y is not None:
            point = np.array([float(x) for x in y.split()], dtype=float)
            return [point]

def calc_angle(p_prev: np.ndarray, p_mid: np.ndarray, p_next: np.ndarray, eps: float = 1e-9) -> float:
    """Calculate angle penalty for path smoothness.
    
    Args:
        p_prev: Previous point coordinates (x, y, z)
        p_mid: Middle point coordinates (x, y, z)
        p_next: Next point coordinates (x, y, z)
        eps: Small epsilon value to avoid division by zero
        
    Returns:
        Angle in radians (0 = straight, pi = reversal)
    """
    v1 = p_prev - p_mid
    v2 = p_next - p_mid
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < eps or n2 < eps:
        return 0.0
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    theta = np.arccos(cosang)
    return theta
