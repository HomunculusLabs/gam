"use client";

import {
  Html,
  OrbitControls,
} from "@react-three/drei";
import {
  Canvas,
  ThreeEvent,
  useFrame,
  useThree,
} from "@react-three/fiber";
import {
  AdditiveBlending,
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CanvasTexture,
  CatmullRomCurve3,
  Color,
  ConeGeometry,
  CylinderGeometry,
  DoubleSide,
  DynamicDrawUsage,
  InstancedMesh,
  MathUtils,
  Object3D,
  Points,
  RepeatWrapping,
  Shape,
  ShapeGeometry,
  SRGBColorSpace,
  Vector3,
} from "three";
import {
  Component,
  ReactNode,
  Suspense,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type Vec3 = [number, number, number];

type District = {
  id: string;
  files: number;
  loc: number;
  tests: number;
  failures: number;
};

type FileBuilding = {
  id: string;
  path: string;
  district: string;
  kind: string;
  loc: number;
  bytes: number;
  tests: number;
  issueRefs: number[];
};

type Issue = {
  number: number;
  title: string;
  state: string;
  comments: number;
  createdAgeDays: number;
  updatedAgeDays: number;
  closedAgeDays: number | null;
  labels: string[];
  url: string;
  refs: number[];
};

type Commit = {
  sha: string;
  fullSha: string;
  date: string;
  subject: string;
  issueRefs: number[];
  files: string[];
  fileCount: number;
  url: string;
};

type Failure = {
  id: string;
  surface: string;
  status: string;
  test: string;
  binary: string;
  file: string | null;
};

type Run = {
  id: number;
  name: string;
  displayTitle: string;
  status: string;
  conclusion: string | null;
  event: string;
  sha: string;
  ageHours: number;
  durationMinutes: number | null;
  url: string;
};

type Measurement = {
  name: string;
  workflow: string;
  status: string;
  runId: string;
  sha: string;
  lagHours: number;
  url: string;
};

type HistorySnapshot = {
  ageDays: number;
  label: string;
  sha: string;
  date: string;
  files: [string, number][];
  fileCount: number;
  bytes: number;
};

type CityData = {
  generatedAt: string;
  repository: {
    name: string;
    url: string;
    defaultBranch: string;
    description: string | null;
  };
  summary: {
    files: number;
    loc: number;
    tests: number;
    failures: number;
    openIssues: number;
    totalIssues: number;
    comments: number;
    commits7d: number;
    runs: number;
    stars: number;
    measurementLagHours: number;
  };
  districts: District[];
  files: FileBuilding[];
  issues: Issue[];
  issueEdges: { source: number; target: number; kind: string }[];
  issueCounts: {
    total: number;
    open: number;
    closed: number;
    comments: number;
  };
  commits: Commit[];
  commitFileEdges: { commit: string; file: string }[];
  commitIssueEdges: { commit: string; issue: number }[];
  failures: Failure[];
  runs: Run[];
  measurements: Measurement[];
  dependencies: { source: string; target: string; weight: number }[];
  history: HistorySnapshot[];
};

type BuildingWorld = FileBuilding & {
  position: Vec3;
  scale: Vec3;
  rotation: number;
  neighborhood: string;
  activity: number;
  failureCount: number;
};

type IssueWorld = Issue & {
  position: Vec3;
  height: number;
  radius: number;
};

type DistrictWorld = District & {
  position: Vec3;
  width: number;
  depth: number;
  top: number;
  centrality: number;
  community: number;
  completedIssues: number;
};

type DistrictConnection = {
  id: string;
  source: string;
  target: string;
  dependency: number;
  coChange: number;
  sharedIssues: number;
  strength: number;
  betweenness: number;
  level: number;
};

type World = {
  districts: DistrictWorld[];
  buildings: BuildingWorld[];
  issues: IssueWorld[];
  buildingById: Map<string, BuildingWorld>;
  issueByNumber: Map<number, IssueWorld>;
  districtById: Map<string, DistrictWorld>;
  connections: DistrictConnection[];
  connectionById: Map<string, DistrictConnection>;
};

type Selection =
  | { kind: "building"; id: string }
  | { kind: "issue"; id: number }
  | { kind: "district"; id: string }
  | { kind: "connection"; id: string }
  | null;

type CameraView = "overview" | "street" | "top";

const tempObject = new Object3D();
const BUILDING_BASE = 0.055;

const BUILDING_COLORS: Record<string, string> = {
  rust: "#d9855d",
  python: "#778bd3",
  test: "#dfa84d",
  workflow: "#4d9fab",
  measurement: "#50a17f",
  documentation: "#b79f82",
  infrastructure: "#8c9eae",
};

const BUILDING_ASPECTS: Record<string, number> = {
  rust: 0.82,
  python: 1.24,
  test: 1.42,
  workflow: 0.7,
  measurement: 1.12,
  documentation: 1.58,
  infrastructure: 0.94,
};

const COMMUNITY_COLORS = [
  "#71917f",
  "#778ba5",
  "#9b8b6f",
  "#837da0",
  "#6f9198",
  "#947d78",
];

function hash(value: string | number) {
  const text = String(value);
  let h = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    h ^= text.charCodeAt(index);
    h = Math.imul(h, 16777619);
  }
  return (h >>> 0) / 4294967295;
}

function compact(value: number) {
  return new Intl.NumberFormat("en", {
    notation: "compact",
    maximumFractionDigits: value >= 1_000_000 ? 2 : 1,
  }).format(value);
}

function ageDays(date: string, referenceDate: string) {
  return Math.max(
    0,
    (new Date(referenceDate).getTime() - new Date(date).getTime()) /
      (24 * 60 * 60 * 1000),
  );
}

function fileNeighborhood(file: FileBuilding) {
  const parts = file.path.split("/");
  if (parts[0] === "crates") return parts.slice(0, 4).join("/");
  if (parts[0] === "tests") return parts.slice(0, 2).join("/");
  if (parts[0] === "bench") return parts.slice(0, 3).join("/");
  return parts.slice(0, 2).join("/");
}

function canonicalFailureFile(file: string | null) {
  if (!file) return null;
  const overlay = "/overlay/";
  const overlayIndex = file.indexOf(overlay);
  return overlayIndex >= 0 ? file.slice(overlayIndex + overlay.length) : file;
}

function snapshotTimestamp(date: Date) {
  return `${date.toISOString().slice(0, 16).replace("T", " ")} UTC`;
}

function districtPair(a: string, b: string) {
  return a < b ? `${a}\u0000${b}` : `${b}\u0000${a}`;
}

function buildDistrictConnections(data: CityData): DistrictConnection[] {
  const fileDistrict = new Map(data.files.map((file) => [file.id, file.district]));
  const values = new Map<
    string,
    Omit<
      DistrictConnection,
      "id" | "source" | "target" | "strength" | "betweenness" | "level"
    >
  >();
  const touch = (
    source: string,
    target: string,
    field: "dependency" | "coChange" | "sharedIssues",
    amount = 1,
  ) => {
    if (source === target) return;
    const key = districtPair(source, target);
    const row = values.get(key) ?? {
      dependency: 0,
      coChange: 0,
      sharedIssues: 0,
    };
    row[field] += amount;
    values.set(key, row);
  };

  data.dependencies.forEach((edge) =>
    touch(edge.source, edge.target, "dependency", edge.weight),
  );

  const commitDistricts = new Map<string, Set<string>>();
  data.commitFileEdges.forEach((edge) => {
    const district = fileDistrict.get(edge.file);
    if (!district) return;
    const districts = commitDistricts.get(edge.commit) ?? new Set<string>();
    districts.add(district);
    commitDistricts.set(edge.commit, districts);
  });
  commitDistricts.forEach((districts) => {
    const rows = [...districts].slice(0, 12);
    rows.forEach((source, index) => {
      rows.slice(index + 1).forEach((target) =>
        touch(source, target, "coChange"),
      );
    });
  });

  const issueDistricts = new Map<number, Set<string>>();
  data.files.forEach((file) => {
    file.issueRefs.forEach((issue) => {
      const districts = issueDistricts.get(issue) ?? new Set<string>();
      districts.add(file.district);
      issueDistricts.set(issue, districts);
    });
  });
  issueDistricts.forEach((districts) => {
    const rows = [...districts].slice(0, 14);
    rows.forEach((source, index) => {
      rows.slice(index + 1).forEach((target) =>
        touch(source, target, "sharedIssues"),
      );
    });
  });

  return [...values.entries()]
    .map(([key, row]) => {
      const [source, target] = key.split("\u0000");
      return {
        id: `${source}::${target}`,
        source,
        target,
        ...row,
        strength:
          row.dependency * 2.4 +
          Math.sqrt(row.coChange) * 1.8 +
          Math.sqrt(row.sharedIssues) * 1.35,
        betweenness: 0,
        level: 0,
      };
    })
    .filter((edge) => edge.strength > 0)
    .sort((a, b) => b.strength - a.strength);
}

function districtCentrality(
  districts: District[],
  connections: DistrictConnection[],
) {
  const ids = districts.map((district) => district.id);
  const rank = new Map(ids.map((id) => [id, 1 / ids.length]));
  const outgoing = new Map(ids.map((id) => [id, 0]));
  connections.forEach((edge) => {
    outgoing.set(edge.source, (outgoing.get(edge.source) ?? 0) + edge.strength);
    outgoing.set(edge.target, (outgoing.get(edge.target) ?? 0) + edge.strength);
  });
  for (let iteration = 0; iteration < 32; iteration += 1) {
    const next = new Map(ids.map((id) => [id, 0.15 / ids.length]));
    connections.forEach((edge) => {
      const sourceShare =
        ((rank.get(edge.source) ?? 0) * edge.strength) /
        Math.max(1, outgoing.get(edge.source) ?? 0);
      const targetShare =
        ((rank.get(edge.target) ?? 0) * edge.strength) /
        Math.max(1, outgoing.get(edge.target) ?? 0);
      next.set(edge.target, (next.get(edge.target) ?? 0) + sourceShare * 0.85);
      next.set(edge.source, (next.get(edge.source) ?? 0) + targetShare * 0.85);
    });
    next.forEach((value, id) => rank.set(id, value));
  }
  return rank;
}

function districtCommunities(
  districts: District[],
  connections: DistrictConnection[],
) {
  const ids = districts.map((district) => district.id).sort();
  const adjacency = new Map(
    ids.map((id) => [id, [] as { neighbor: string; strength: number }[]]),
  );
  connections.forEach((edge) => {
    adjacency.get(edge.source)?.push({
      neighbor: edge.target,
      strength: edge.strength,
    });
    adjacency.get(edge.target)?.push({
      neighbor: edge.source,
      strength: edge.strength,
    });
  });
  const degree = new Map(
    ids.map((id) => [
      id,
      (adjacency.get(id) ?? []).reduce(
        (sum, edge) => sum + edge.strength,
        0,
      ),
    ]),
  );
  const totalDegree = Math.max(
    1,
    [...degree.values()].reduce((sum, value) => sum + value, 0),
  );
  const label = new Map(ids.map((id) => [id, id]));
  const communityDegree = new Map(ids.map((id) => [id, degree.get(id) ?? 0]));

  for (let pass = 0; pass < 18; pass += 1) {
    let moved = false;
    ids.forEach((id) => {
      const current = label.get(id)!;
      const nodeDegree = degree.get(id) ?? 0;
      communityDegree.set(
        current,
        (communityDegree.get(current) ?? 0) - nodeDegree,
      );
      const internal = new Map<string, number>();
      (adjacency.get(id) ?? []).forEach((edge) => {
        const community = label.get(edge.neighbor)!;
        internal.set(
          community,
          (internal.get(community) ?? 0) + edge.strength,
        );
      });
      internal.set(current, internal.get(current) ?? 0);
      let best = current;
      let bestGain = Number.NEGATIVE_INFINITY;
      [...internal.keys()].sort().forEach((candidate) => {
        const gain =
          (internal.get(candidate) ?? 0) -
          (nodeDegree * (communityDegree.get(candidate) ?? 0)) / totalDegree;
        if (gain > bestGain + 1e-9) {
          best = candidate;
          bestGain = gain;
        }
      });
      label.set(id, best);
      communityDegree.set(
        best,
        (communityDegree.get(best) ?? 0) + nodeDegree,
      );
      if (best !== current) moved = true;
    });
    if (!moved) break;
  }

  const labels = [...new Set(label.values())].sort();
  const index = new Map(labels.map((value, position) => [value, position]));
  return new Map(ids.map((id) => [id, index.get(label.get(id)!) ?? 0]));
}

function districtSlots(
  districts: District[],
  connections: DistrictConnection[],
): Map<string, Vec3> {
  const centrality = districtCentrality(districts, connections);
  const communities = districtCommunities(districts, connections);
  const ordered = [...districts].sort(
    (a, b) => (centrality.get(b.id) ?? 0) - (centrality.get(a.id) ?? 0),
  );
  const positions = new Map<string, { x: number; z: number }>();
  const goldenAngle = Math.PI * (3 - Math.sqrt(5));

  ordered.forEach((district, index) => {
    if (index === 0) {
      positions.set(district.id, { x: 0, z: 0 });
      return;
    }
    const radius = 34 * Math.sqrt(index);
    const angle = goldenAngle * index + Math.sin(index * 2.17) * 0.18;
    positions.set(district.id, {
      x: Math.cos(angle) * radius,
      z: Math.sin(angle) * radius * 0.78,
    });
  });

  for (let iteration = 0; iteration < 180; iteration += 1) {
    const force = new Map(
      ordered.map((district) => [district.id, { x: 0, z: 0 }]),
    );
    ordered.forEach((a, index) => {
      ordered.slice(index + 1).forEach((b) => {
        const pa = positions.get(a.id)!;
        const pb = positions.get(b.id)!;
        const dx = pa.x - pb.x;
        const dz = pa.z - pb.z;
        const distanceSquared = Math.max(80, dx * dx + dz * dz);
        const push = 2600 / distanceSquared;
        const distance = Math.sqrt(distanceSquared);
        force.get(a.id)!.x += (dx / distance) * push;
        force.get(a.id)!.z += (dz / distance) * push;
        force.get(b.id)!.x -= (dx / distance) * push;
        force.get(b.id)!.z -= (dz / distance) * push;
      });
    });
    connections.slice(0, 90).forEach((edge) => {
      const source = positions.get(edge.source);
      const target = positions.get(edge.target);
      if (!source || !target) return;
      const dx = target.x - source.x;
      const dz = target.z - source.z;
      const distance = Math.max(1, Math.hypot(dx, dz));
      const sameCommunity =
        communities.get(edge.source) === communities.get(edge.target);
      const ideal =
        (sameCommunity ? 19 : 31) + 28 / Math.log2(edge.strength + 2);
      const pull = (distance - ideal) * Math.min(0.025, edge.strength * 0.0015);
      force.get(edge.source)!.x += (dx / distance) * pull;
      force.get(edge.source)!.z += (dz / distance) * pull;
      force.get(edge.target)!.x -= (dx / distance) * pull;
      force.get(edge.target)!.z -= (dz / distance) * pull;
    });
    const cooling = 0.42 * (1 - iteration / 220);
    ordered.forEach((district) => {
      const point = positions.get(district.id)!;
      const delta = force.get(district.id)!;
      point.x += (delta.x - point.x * 0.0025) * cooling;
      point.z += (delta.z - point.z * 0.0025) * cooling;
    });
  }

  const maxRadius = Math.max(
    ...[...positions.values()].map((point) => Math.hypot(point.x, point.z)),
  );
  const scale = 145 / Math.max(1, maxRadius);
  return new Map(
    [...positions].map(([id, point]) => [
      id,
      [point.x * scale, 0, point.z * scale] as Vec3,
    ]),
  );
}

function roadBetweenness(
  districts: DistrictWorld[],
  connections: DistrictConnection[],
) {
  const ids = districts.map((district) => district.id);
  const adjacency = new Map(
    ids.map((id) => [
      id,
      [] as { neighbor: string; edgeId: string; cost: number }[],
    ]),
  );
  connections.forEach((edge) => {
    const cost = 1 / Math.max(0.001, edge.strength);
    adjacency.get(edge.source)?.push({
      neighbor: edge.target,
      edgeId: edge.id,
      cost,
    });
    adjacency.get(edge.target)?.push({
      neighbor: edge.source,
      edgeId: edge.id,
      cost,
    });
  });
  const score = new Map(connections.map((edge) => [edge.id, 0]));

  ids.forEach((source) => {
    const distance = new Map(ids.map((id) => [id, Number.POSITIVE_INFINITY]));
    const paths = new Map(ids.map((id) => [id, 0]));
    const predecessors = new Map(
      ids.map((id) => [
        id,
        [] as { node: string; edgeId: string }[],
      ]),
    );
    const visited = new Set<string>();
    const stack: string[] = [];
    distance.set(source, 0);
    paths.set(source, 1);

    while (visited.size < ids.length) {
      const current = ids
        .filter((id) => !visited.has(id))
        .sort(
          (a, b) =>
            (distance.get(a) ?? Number.POSITIVE_INFINITY) -
            (distance.get(b) ?? Number.POSITIVE_INFINITY),
        )[0];
      if (
        !current ||
        !Number.isFinite(distance.get(current) ?? Number.POSITIVE_INFINITY)
      ) {
        break;
      }
      visited.add(current);
      stack.push(current);
      (adjacency.get(current) ?? []).forEach((edge) => {
        const candidate = (distance.get(current) ?? 0) + edge.cost;
        const known = distance.get(edge.neighbor) ?? Number.POSITIVE_INFINITY;
        if (candidate < known - 1e-9) {
          distance.set(edge.neighbor, candidate);
          paths.set(edge.neighbor, paths.get(current) ?? 0);
          predecessors.set(edge.neighbor, [
            { node: current, edgeId: edge.edgeId },
          ]);
        } else if (Math.abs(candidate - known) <= 1e-9) {
          paths.set(
            edge.neighbor,
            (paths.get(edge.neighbor) ?? 0) + (paths.get(current) ?? 0),
          );
          predecessors
            .get(edge.neighbor)
            ?.push({ node: current, edgeId: edge.edgeId });
        }
      });
    }

    const dependency = new Map(ids.map((id) => [id, 0]));
    stack.reverse().forEach((node) => {
      (predecessors.get(node) ?? []).forEach((predecessor) => {
        const contribution =
          ((paths.get(predecessor.node) ?? 0) /
            Math.max(1, paths.get(node) ?? 0)) *
          (1 + (dependency.get(node) ?? 0));
        score.set(
          predecessor.edgeId,
          (score.get(predecessor.edgeId) ?? 0) + contribution,
        );
        dependency.set(
          predecessor.node,
          (dependency.get(predecessor.node) ?? 0) + contribution,
        );
      });
    });
  });

  const maximum = Math.max(1, ...score.values());
  return new Map(
    [...score].map(([edgeId, value]) => [edgeId, value / maximum]),
  );
}

function roadSegmentsCross(
  sourceA: DistrictWorld,
  targetA: DistrictWorld,
  sourceB: DistrictWorld,
  targetB: DistrictWorld,
) {
  const orientation = (
    a: DistrictWorld,
    b: DistrictWorld,
    c: DistrictWorld,
  ) =>
    (b.position[0] - a.position[0]) * (c.position[2] - a.position[2]) -
    (b.position[2] - a.position[2]) * (c.position[0] - a.position[0]);
  const abC = orientation(sourceA, targetA, sourceB);
  const abD = orientation(sourceA, targetA, targetB);
  const cdA = orientation(sourceB, targetB, sourceA);
  const cdB = orientation(sourceB, targetB, targetA);
  return abC * abD < 0 && cdA * cdB < 0;
}

function gradeRoadNetwork(
  districts: DistrictWorld[],
  connections: DistrictConnection[],
) {
  const districtById = new Map(
    districts.map((district) => [district.id, district]),
  );
  const betweenness = roadBetweenness(districts, connections);
  const ranked = connections
    .map((edge) => ({
      ...edge,
      betweenness: betweenness.get(edge.id) ?? 0,
      level: 0,
    }))
    .sort(
      (a, b) =>
        b.betweenness * 3 +
        Math.log2(b.strength + 1) -
        (a.betweenness * 3 + Math.log2(a.strength + 1)),
    );
  const placed: DistrictConnection[] = [];
  ranked.forEach((edge) => {
    const source = districtById.get(edge.source);
    const target = districtById.get(edge.target);
    if (!source || !target) return;
    const occupied = new Set<number>();
    placed.forEach((other) => {
      if (
        edge.source === other.source ||
        edge.source === other.target ||
        edge.target === other.source ||
        edge.target === other.target
      ) {
        return;
      }
      const otherSource = districtById.get(other.source);
      const otherTarget = districtById.get(other.target);
      if (
        otherSource &&
        otherTarget &&
        roadSegmentsCross(source, target, otherSource, otherTarget)
      ) {
        occupied.add(other.level);
      }
    });
    let level = 0;
    while (occupied.has(level)) level += 1;
    edge.level = level;
    placed.push(edge);
  });
  const byId = new Map(placed.map((edge) => [edge.id, edge]));
  return connections.map((edge) => byId.get(edge.id) ?? edge);
}

function makeWorld(data: CityData): World {
  const connections = buildDistrictConnections(data);
  const centrality = districtCentrality(data.districts, connections);
  const communities = districtCommunities(data.districts, connections);
  const slots = districtSlots(data.districts, connections);
  const commitsByFile = new CounterMap<string>();
  const recentCutoff = 30;
  const commitBySha = new Map(data.commits.map((commit) => [commit.sha, commit]));

  for (const edge of data.commitFileEdges) {
    const commit = commitBySha.get(edge.commit);
    if (commit && ageDays(commit.date, data.generatedAt) <= recentCutoff) {
      commitsByFile.increment(edge.file);
    }
  }

  const failuresByFile = new CounterMap<string>();
  for (const failure of data.failures) {
    const file = canonicalFailureFile(failure.file);
    if (file) failuresByFile.increment(file);
  }

  const districtRows = new Map<string, FileBuilding[]>();
  for (const file of data.files) {
    const list = districtRows.get(file.district) ?? [];
    list.push(file);
    districtRows.set(file.district, list);
  }

  const closedIssues = new Set(
    data.issues
      .filter((issue) => issue.state === "closed")
      .map((issue) => issue.number),
  );
  const completedByDistrict = new Map<string, Set<number>>();
  data.files.forEach((file) => {
    file.issueRefs.forEach((issue) => {
      if (!closedIssues.has(issue)) return;
      const completed =
        completedByDistrict.get(file.district) ?? new Set<number>();
      completed.add(issue);
      completedByDistrict.set(file.district, completed);
    });
  });

  const districts: DistrictWorld[] = data.districts.map((district) => {
    const rows = districtRows.get(district.id) ?? [];
    const radius = Math.sqrt(rows.length) * 0.68 + 5;
    const meanPathDepth =
      rows.reduce((sum, file) => sum + file.path.split("/").length, 0) /
      Math.max(1, rows.length);
    const aspect = MathUtils.clamp(
      0.78 + (meanPathDepth - 3) * 0.08,
      0.76,
      1.24,
    );
    const width = Math.max(11, Math.min(45, radius * 2 * aspect));
    const depth = Math.max(11, Math.min(42, (radius * 2) / aspect));
    return {
      ...district,
      position: slots.get(district.id) ?? [0, 0, 0],
      width,
      depth,
      top: 0.18,
      centrality: centrality.get(district.id) ?? 0,
      community: communities.get(district.id) ?? 0,
      completedIssues: completedByDistrict.get(district.id)?.size ?? 0,
    };
  });
  const districtById = new Map(districts.map((district) => [district.id, district]));
  const gradedConnections = gradeRoadNetwork(districts, connections);

  const buildings: BuildingWorld[] = [];
  for (const district of districts) {
    const rows = (districtRows.get(district.id) ?? []).sort(
      (a, b) => b.loc - a.loc,
    );
    const strongestConnection = connections.find(
      (connection) =>
        connection.source === district.id || connection.target === district.id,
    );
    const connectedDistrictId = strongestConnection
      ? strongestConnection.source === district.id
        ? strongestConnection.target
        : strongestConnection.source
      : null;
    const connectedPosition = connectedDistrictId
      ? slots.get(connectedDistrictId)
      : null;
    const districtTurn = connectedPosition
      ? Math.atan2(
          connectedPosition[2] - district.position[2],
          connectedPosition[0] - district.position[0],
        )
      : 0;
    const neighborhoods = new Map<string, FileBuilding[]>();
    rows.forEach((file) => {
      const key = fileNeighborhood(file);
      const group = neighborhoods.get(key) ?? [];
      group.push(file);
      neighborhoods.set(key, group);
    });
    const neighborhoodRows = [...neighborhoods.entries()].sort(
      (a, b) =>
        b[1].reduce((sum, file) => sum + file.loc, 0) -
        a[1].reduce((sum, file) => sum + file.loc, 0),
    );
    const layout = new Map<
      string,
      { x: number; z: number; theta: number; neighborhood: string }
    >();
    const neighborhoodByFile = new Map(
      rows.map((file) => [file.id, fileNeighborhood(file)]),
    );
    const neighborhoodLinks = new Map<string, number>();
    const addNeighborhoodLinks = (members: Set<string>, weight: number) => {
      const values = [...members].sort();
      values.forEach((source, index) => {
        values.slice(index + 1).forEach((target) => {
          const key = districtPair(source, target);
          neighborhoodLinks.set(
            key,
            (neighborhoodLinks.get(key) ?? 0) + weight,
          );
        });
      });
    };
    const commitNeighborhoods = new Map<string, Set<string>>();
    data.commitFileEdges.forEach((edge) => {
      const neighborhood = neighborhoodByFile.get(edge.file);
      if (!neighborhood) return;
      const members = commitNeighborhoods.get(edge.commit) ?? new Set<string>();
      members.add(neighborhood);
      commitNeighborhoods.set(edge.commit, members);
    });
    commitNeighborhoods.forEach((members) => addNeighborhoodLinks(members, 2));
    const issueNeighborhoods = new Map<number, Set<string>>();
    rows.forEach((file) => {
      const neighborhood = neighborhoodByFile.get(file.id)!;
      file.issueRefs.forEach((issue) => {
        const members = issueNeighborhoods.get(issue) ?? new Set<string>();
        members.add(neighborhood);
        issueNeighborhoods.set(issue, members);
      });
    });
    issueNeighborhoods.forEach((members) => addNeighborhoodLinks(members, 1));

    const centers: {
      neighborhood: string;
      files: FileBuilding[];
      x: number;
      z: number;
      radius: number;
      angle: number;
    }[] = [];
    neighborhoodRows.forEach(([neighborhood, files], neighborhoodIndex) => {
      const clusterRadius = Math.min(
        Math.max(1.4, Math.sqrt(files.length) * 0.48),
        Math.min(district.width, district.depth) * 0.2,
      );
      if (neighborhoodIndex === 0) {
        centers.push({
          neighborhood,
          files,
          x: 0,
          z: 0,
          radius: clusterRadius,
          angle: districtTurn,
        });
        return;
      }
      const parent = centers.reduce((best, candidate) => {
        const candidateLink =
          neighborhoodLinks.get(
            districtPair(neighborhood, candidate.neighborhood),
          ) ?? 0;
        const bestLink =
          neighborhoodLinks.get(districtPair(neighborhood, best.neighborhood)) ??
          0;
        if (candidateLink !== bestLink) {
          return candidateLink > bestLink ? candidate : best;
        }
        const candidatePrefix = candidate.neighborhood
          .split("/")
          .filter(
            (part, index) => neighborhood.split("/")[index] === part,
          ).length;
        const bestPrefix = best.neighborhood
          .split("/")
          .filter(
            (part, index) => neighborhood.split("/")[index] === part,
          ).length;
        return candidatePrefix > bestPrefix ? candidate : best;
      }, centers[0]);
      const linkStrength =
        neighborhoodLinks.get(districtPair(neighborhood, parent.neighborhood)) ??
        0;
      const lexicalSide =
        neighborhood.localeCompare(parent.neighborhood) >= 0 ? 1 : -1;
      const angle =
        parent.angle +
        lexicalSide * (0.48 + 0.72 / Math.sqrt(linkStrength + 1));
      const distance =
        (parent.radius + clusterRadius) * 0.74 +
        1.8 / Math.sqrt(linkStrength + 1);
      centers.push({
        neighborhood,
        files,
        x: parent.x + Math.cos(angle) * distance,
        z: parent.z + Math.sin(angle) * distance,
        radius: clusterRadius,
        angle,
      });
    });
    for (let iteration = 0; iteration < 18; iteration += 1) {
      centers.forEach((source, index) => {
        centers.slice(index + 1).forEach((target) => {
          const dx = target.x - source.x;
          const dz = target.z - source.z;
          const distance = Math.max(0.05, Math.hypot(dx, dz));
          const minimum = (source.radius + target.radius) * 0.66;
          if (distance >= minimum) return;
          const shift = (minimum - distance) * 0.22;
          source.x -= (dx / distance) * shift;
          source.z -= (dz / distance) * shift;
          target.x += (dx / distance) * shift;
          target.z += (dz / distance) * shift;
        });
      });
    }
    const centerScale = Math.min(
      1,
      ...centers.flatMap((center) => [
        (district.width * 0.42) / Math.max(0.1, Math.abs(center.x) + center.radius),
        (district.depth * 0.42) / Math.max(0.1, Math.abs(center.z) + center.radius),
      ]),
    );
    centers.forEach((center) => {
      const centerX = center.x * centerScale;
      const centerZ = center.z * centerScale;
      const files = [...center.files].sort((a, b) =>
        a.path.localeCompare(b.path),
      );
      const lanes = Math.min(3, Math.max(1, Math.ceil(Math.sqrt(files.length) / 2)));
      const columns = Math.ceil(files.length / lanes);
      const extent = Math.min(
        center.radius * 1.7,
        Math.max(0.6, (columns - 1) * 0.58),
      );
      files.forEach((file, fileIndex) => {
        const lane = fileIndex % lanes;
        const column = Math.floor(fileIndex / lanes);
        const progress = columns <= 1 ? 0.5 : column / (columns - 1);
        const along = (progress - 0.5) * extent * 2;
        const across =
          (lane - (lanes - 1) / 2) * 0.48 +
          Math.sin(progress * Math.PI * 2) * center.radius * 0.16;
        const directionX = Math.cos(center.angle);
        const directionZ = Math.sin(center.angle);
        const perpendicularX = -directionZ;
        const perpendicularZ = directionX;
        layout.set(file.id, {
          x: centerX + directionX * along + perpendicularX * across,
          z: centerZ + directionZ * along + perpendicularZ * across,
          theta: center.angle,
          neighborhood: center.neighborhood,
        });
      });
    });
    rows.forEach((file) => {
      const activity = commitsByFile.get(file.id);
      const failureCount = failuresByFile.get(file.id);
      const baseHeight = 0.34 + Math.log2(file.loc + 1) * 0.62;
      const height = baseHeight;
      const footprint = Math.max(
        0.3,
        Math.min(
          1.18,
          0.24 +
            Math.log10(file.bytes + 8) * 0.1 +
            Math.log2(file.issueRefs.length + activity + 1) * 0.075,
        ),
      );
      const address = layout.get(file.id)!;
      const theta = address.theta;
      const x = district.position[0] + address.x;
      const z = district.position[2] + address.z;
      const rectangularity =
        (BUILDING_ASPECTS[file.kind] ?? 1) +
        Math.min(0.24, Math.log2(file.tests + 1) * 0.025);
      const rotation = theta + Math.PI / 2;
      buildings.push({
        ...file,
        position: [x, height / 2 + BUILDING_BASE, z],
        scale: [footprint * rectangularity, height, footprint / rectangularity],
        rotation,
        neighborhood: address.neighborhood,
        activity,
        failureCount,
      });
    });
  }
  const buildingById = new Map(buildings.map((building) => [building.id, building]));

  const issueFiles = new Map<number, BuildingWorld[]>();
  for (const building of buildings) {
    for (const issue of building.issueRefs) {
      const linked = issueFiles.get(issue) ?? [];
      linked.push(building);
      issueFiles.set(issue, linked);
    }
  }

  const issues: IssueWorld[] = data.issues.map((issue, index) => {
    const linked = issueFiles.get(issue.number) ?? [];
    let x: number;
    let z: number;
    if (linked.length > 0) {
      const selected = linked.slice(0, 12);
      x = selected.reduce((sum, item) => sum + item.position[0], 0) / selected.length;
      z = selected.reduce((sum, item) => sum + item.position[2], 0) / selected.length;
      const jitter = 1.4 + Math.log2(linked.length + 1);
      x += (hash(issue.number) - 0.5) * jitter;
      z += (hash(`${issue.number}-issue`) - 0.5) * jitter;
    } else {
      const angle = hash(issue.number) * Math.PI * 2;
      const ring = 96 + (index % 5) * 4.2;
      x = Math.cos(angle) * ring;
      z = Math.sin(angle) * ring;
    }
    const ageMass = Math.min(3.2, issue.createdAgeDays / 420);
    const commentMass = Math.log2(issue.comments + 1);
    return {
      ...issue,
      position: [x, 0.25, z],
      height:
        issue.state === "open"
          ? 2.4 + commentMass * 1.5 + ageMass
          : 0.18,
      radius: Math.max(0.48, Math.min(1.7, 0.45 + commentMass * 0.2)),
    };
  });
  const issueByNumber = new Map(issues.map((issue) => [issue.number, issue]));

  return {
    districts,
    buildings,
    issues,
    buildingById,
    issueByNumber,
    districtById,
    connections: gradedConnections,
    connectionById: new Map(
      gradedConnections.map((connection) => [connection.id, connection]),
    ),
  };
}

class CounterMap<T> {
  private values = new Map<T, number>();

  increment(key: T) {
    this.values.set(key, (this.values.get(key) ?? 0) + 1);
  }

  get(key: T) {
    return this.values.get(key) ?? 0;
  }
}

class SceneErrorBoundary extends Component<
  { children: ReactNode },
  { failed: boolean }
> {
  state = { failed: false };

  static getDerivedStateFromError() {
    return { failed: true };
  }

  render() {
    if (this.state.failed) {
      return (
        <div className="loading-fallback">
          The city renderer could not start on this graphics device.
        </div>
      );
    }
    return this.props.children;
  }
}

function CameraRig({ view }: { view: CameraView }) {
  const { camera } = useThree();
  const target = useRef(new Vector3(0, 3, 0));
  const origin = useRef(camera.position.clone());
  const progress = useRef(1);
  const desired = useMemo(() => {
    if (view === "street") return new Vector3(54, 14, 58);
    if (view === "top") return new Vector3(0, 150, 0.01);
    return new Vector3(90, 72, 96);
  }, [view]);

  useEffect(() => {
    origin.current.copy(camera.position);
    progress.current = 0;
  }, [camera, desired]);

  useFrame((_, delta) => {
    if (progress.current >= 1) return;
    progress.current = Math.min(1, progress.current + delta * 1.35);
    const eased = 1 - Math.pow(1 - progress.current, 3);
    camera.position.lerpVectors(origin.current, desired, eased);
    camera.lookAt(target.current);
  });
  return null;
}

function BuildingGroup({
  buildings,
  kind,
  color,
  onSelect,
  historyFiles,
}: {
  buildings: BuildingWorld[];
  kind: string;
  color: string;
  onSelect: (selection: Selection) => void;
  historyFiles: Map<string, number>;
}) {
  const meshRef = useRef<InstancedMesh>(null);
  const capRef = useRef<InstancedMesh>(null);
  const bandRef = useRef<InstancedMesh>(null);
  const geometry = useMemo(() => new BoxGeometry(1, 1, 1), []);
  const facadeTexture = useMemo(() => {
    const canvas = document.createElement("canvas");
    canvas.width = 64;
    canvas.height = 128;
    const context = canvas.getContext("2d");
    if (!context) throw new Error("Canvas textures are unavailable");
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = "#b8c4c3";
    context.strokeStyle = "#9dacab";
    context.lineWidth = 2;

    if (kind === "rust" || kind === "documentation") {
      for (let y = 8; y < canvas.height; y += kind === "rust" ? 10 : 14) {
        context.fillRect(0, y, canvas.width, 2);
        const offset = (y / 2) % 16;
        for (let x = offset; x < canvas.width; x += 16) {
          context.fillRect(x, y - 8, 1, 8);
        }
      }
    } else if (kind === "python" || kind === "measurement") {
      for (let x = 6; x < canvas.width; x += kind === "python" ? 12 : 9) {
        context.fillRect(x, 0, 2, canvas.height);
      }
      for (let y = 12; y < canvas.height; y += 18) {
        context.fillRect(0, y, canvas.width, 1);
      }
    } else if (kind === "test") {
      for (let y = 8; y < canvas.height; y += 12) {
        for (let x = (y / 12) % 2 === 0 ? 4 : 12; x < canvas.width; x += 16) {
          context.fillRect(x, y, 8, 3);
        }
      }
    } else if (kind === "workflow") {
      for (let offset = -128; offset < 128; offset += 14) {
        context.beginPath();
        context.moveTo(offset, 0);
        context.lineTo(offset + 128, 128);
        context.stroke();
      }
    } else {
      for (let y = 10; y < canvas.height; y += 16) {
        context.fillRect(0, y, canvas.width, 2);
      }
    }
    const texture = new CanvasTexture(canvas);
    texture.colorSpace = SRGBColorSpace;
    texture.wrapS = RepeatWrapping;
    texture.wrapT = RepeatWrapping;
    texture.repeat.set(1, 3);
    return texture;
  }, [kind]);
  const capColor = useMemo(
    () => `#${new Color(color).lerp(new Color("#f5ffff"), 0.38).getHexString()}`,
    [color],
  );
  const bandColor = useMemo(
    () => `#${new Color(color).lerp(new Color("#26383c"), 0.34).getHexString()}`,
    [color],
  );
  useEffect(
    () => () => {
      facadeTexture.dispose();
    },
    [facadeTexture],
  );
  useEffect(
    () => () => {
      geometry.dispose();
    },
    [geometry],
  );

  useLayoutEffect(() => {
    const mesh = meshRef.current;
    const caps = capRef.current;
    const bands = bandRef.current;
    if (!mesh || !caps || !bands) return;
    buildings.forEach((building, index) => {
      const historicBytes = historyFiles.get(building.id);
      const exists = historicBytes !== undefined;
      const massRatio = exists
        ? Math.max(0.12, Math.min(3.2, historicBytes / Math.max(1, building.bytes)))
        : 0.0001;
      const historicScale = exists ? Math.sqrt(massRatio) : 0.0001;
      const historicHeight = building.scale[1] * historicScale;
      tempObject.position.set(
        building.position[0],
        BUILDING_BASE + historicHeight / 2,
        building.position[2],
      );
      tempObject.rotation.set(0, building.rotation, 0);
      tempObject.scale.set(
        building.scale[0] * historicScale,
        historicHeight,
        building.scale[2] * historicScale,
      );
      tempObject.updateMatrix();
      mesh.setMatrixAt(index, tempObject.matrix);

      const tall =
        exists &&
        building.scale[1] > 2.8 &&
        (building.tests > 0 || building.activity > 0);
      const capHeight =
        Math.min(
          2.6,
          0.14 +
            Math.log2(building.tests + 1) * 0.12 +
            building.activity * 0.09,
        ) * historicScale;
      tempObject.position.set(
        building.position[0],
        BUILDING_BASE + historicHeight + capHeight / 2,
        building.position[2],
      );
      tempObject.scale.set(
        tall ? building.scale[0] * 0.62 * historicScale : 0.0001,
        tall ? capHeight : 0.0001,
        tall ? building.scale[2] * 0.62 * historicScale : 0.0001,
      );
      tempObject.updateMatrix();
      caps.setMatrixAt(index, tempObject.matrix);

      const bandHeight =
        exists && historicHeight > 0.5
          ? Math.min(0.14, 0.055 + historicHeight * 0.008)
          : 0.0001;
      const bandLevel =
        0.4 + Math.min(0.32, Math.log2(building.tests + 1) * 0.038);
      tempObject.position.set(
        building.position[0],
        BUILDING_BASE + historicHeight * bandLevel,
        building.position[2],
      );
      tempObject.scale.set(
        exists ? building.scale[0] * historicScale * 1.035 : 0.0001,
        bandHeight,
        exists ? building.scale[2] * historicScale * 1.035 : 0.0001,
      );
      tempObject.updateMatrix();
      bands.setMatrixAt(index, tempObject.matrix);
    });
    mesh.instanceMatrix.setUsage(DynamicDrawUsage);
    mesh.instanceMatrix.needsUpdate = true;
    caps.instanceMatrix.setUsage(DynamicDrawUsage);
    caps.instanceMatrix.needsUpdate = true;
    bands.instanceMatrix.setUsage(DynamicDrawUsage);
    bands.instanceMatrix.needsUpdate = true;
  }, [buildings, historyFiles]);

  return (
    <>
      <instancedMesh
        ref={meshRef}
        args={[geometry, undefined, buildings.length]}
        onClick={(event: ThreeEvent<MouseEvent>) => {
          event.stopPropagation();
          if (event.instanceId === undefined) return;
          const building = buildings[event.instanceId];
          onSelect({ kind: "building", id: building.id });
        }}
        onPointerOver={() => {
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "";
        }}
      >
        <meshBasicMaterial
          color={color}
          map={facadeTexture}
          toneMapped={false}
        />
      </instancedMesh>
      <instancedMesh
        ref={capRef}
        args={[geometry, undefined, buildings.length]}
      >
        <meshBasicMaterial color={capColor} toneMapped={false} />
      </instancedMesh>
      <instancedMesh
        ref={bandRef}
        args={[geometry, undefined, buildings.length]}
      >
        <meshBasicMaterial color={bandColor} toneMapped={false} />
      </instancedMesh>
    </>
  );
}

function Buildings({
  world,
  selection,
  onSelect,
  historyFiles,
}: {
  world: World;
  selection: Selection;
  onSelect: (selection: Selection) => void;
  historyFiles: Map<string, number>;
}) {
  const groups = useMemo(() => {
    const result = new Map<string, BuildingWorld[]>();
    world.buildings.forEach((building) => {
      const kind = BUILDING_COLORS[building.kind]
        ? building.kind
        : "infrastructure";
      const rows = result.get(kind) ?? [];
      rows.push(building);
      result.set(kind, rows);
    });
    return [...result.entries()];
  }, [world]);
  const selected =
    selection?.kind === "building"
      ? world.buildingById.get(selection.id)
      : undefined;
  const selectedExists = selected ? historyFiles.has(selected.id) : false;
  const selectedRatio =
    selected && selectedExists
      ? Math.sqrt(
          Math.max(
            0.12,
            Math.min(
              3.2,
              (historyFiles.get(selected.id) ?? selected.bytes) /
                Math.max(1, selected.bytes),
            ),
          ),
        )
      : 1;

  return (
    <>
      {groups.map(([kind, buildings]) => (
        <BuildingGroup
          key={kind}
          buildings={buildings}
          kind={kind}
          color={BUILDING_COLORS[kind]}
          onSelect={onSelect}
          historyFiles={historyFiles}
        />
      ))}
      {selected && selectedExists ? (
        <mesh
          position={[selected.position[0], 0.046, selected.position[2]]}
          rotation={[-Math.PI / 2, 0, selected.rotation]}
          scale={[
            selected.scale[0] * selectedRatio * 1.7,
            selected.scale[2] * selectedRatio * 1.7,
            1,
          ]}
        >
          <ringGeometry args={[0.72, 1, 32]} />
          <meshBasicMaterial
            color="#fff1a8"
            transparent
            opacity={0.96}
            side={DoubleSide}
            toneMapped={false}
          />
        </mesh>
      ) : null}
    </>
  );
}

function OrganicDistrictParcel({
  district,
  buildings,
  color,
  onSelect,
}: {
  district: DistrictWorld;
  buildings: BuildingWorld[];
  color: Color;
  onSelect: (selection: Selection) => void;
}) {
  const geometry = useMemo(() => {
    const binCount = 28;
    const support = Array.from({ length: binCount }, (_, bin) => {
      const angle = (bin / binCount) * Math.PI * 2;
      const nx = Math.cos(angle);
      const nz = Math.sin(angle);
      return Math.max(
        2.8,
        ...buildings.map((building) => {
          const dx = building.position[0] - district.position[0];
          const dz = building.position[2] - district.position[2];
          const halfFootprint =
            Math.max(building.scale[0], building.scale[2]) * 0.72;
          return dx * nx + dz * nz + halfFootprint + 1.25;
        }),
      );
    });
    const smoothed = support.map(
      (_, index) =>
        (support[(index + binCount - 2) % binCount] +
          support[(index + binCount - 1) % binCount] * 2 +
          support[index] * 3 +
          support[(index + 1) % binCount] * 2 +
          support[(index + 2) % binCount]) /
        9,
    );
    const shape = new Shape();
    smoothed.forEach((radius, index) => {
      const angle = (index / binCount) * Math.PI * 2;
      const x = Math.cos(angle) * radius;
      const z = Math.sin(angle) * radius;
      if (index === 0) shape.moveTo(x, z);
      else shape.lineTo(x, z);
    });
    shape.closePath();
    return new ShapeGeometry(shape);
  }, [buildings, district]);
  useEffect(
    () => () => {
      geometry.dispose();
    },
    [geometry],
  );

  return (
    <mesh
      geometry={geometry}
      position={[district.position[0], 0.012, district.position[2]]}
      rotation={[-Math.PI / 2, 0, 0]}
      onClick={(event) => {
        event.stopPropagation();
        onSelect({ kind: "district", id: district.id });
      }}
    >
      <meshStandardMaterial
        color={color}
        roughness={0.97}
        metalness={0.01}
        transparent
        opacity={0.92}
        side={DoubleSide}
      />
    </mesh>
  );
}

function Districts({
  world,
  onSelect,
  historyFiles,
}: {
  world: World;
  onSelect: (selection: Selection) => void;
  historyFiles: Map<string, number>;
}) {
  return (
    <>
      {world.districts.map((district) => {
        const active = world.buildings.filter(
          (building) =>
            building.district === district.id && historyFiles.has(building.id),
        );
        const activeBuildings = active.length;
        const failurePressure = Math.min(
          1,
          district.failures / Math.max(1, district.tests) * 8,
        );
        const hue = new Color(
          COMMUNITY_COLORS[district.community % COMMUNITY_COLORS.length],
        )
          .lerp(new Color("#ad867d"), failurePressure)
          .lerp(
            new Color("#a9c6bf"),
            Math.min(0.32, district.centrality * 1.8),
          );
        return (
          <group key={district.id}>
            <OrganicDistrictParcel
              district={district}
              buildings={active}
              color={hue}
              onSelect={onSelect}
            />
            <Html
              position={[
                district.position[0] - district.width / 2,
                1.1,
                district.position[2] - district.depth / 2,
              ]}
              distanceFactor={34}
              zIndexRange={[8, 0]}
            >
              <div className="district-label">
                <strong>{district.id}</strong>
                <span>
                  {activeBuildings}/{district.files} buildings · {district.failures} fires
                </span>
              </div>
            </Html>
          </group>
        );
      })}
    </>
  );
}

function SurfaceRibbon({
  points,
  width,
  color,
  opacity = 1,
  layer = 0,
  onClick,
}: {
  points: Vec3[];
  width: number;
  color: string;
  opacity?: number;
  layer?: number;
  onClick?: (event: ThreeEvent<MouseEvent>) => void;
}) {
  const geometry = useMemo(() => {
    const curve = new CatmullRomCurve3(
      points.map(([x, y, z]) => new Vector3(x, y, z)),
      false,
      "catmullrom",
      0.35,
    );
    const segments = Math.max(5, Math.min(24, (points.length - 1) * 6));
    const positions: number[] = [];
    const uvs: number[] = [];
    const indices: number[] = [];
    for (let index = 0; index <= segments; index += 1) {
      const t = index / segments;
      const point = curve.getPoint(t);
      const tangent = curve.getTangent(t);
      const perpendicular = new Vector3(-tangent.z, 0, tangent.x).normalize();
      const left = point.clone().addScaledVector(perpendicular, width / 2);
      const right = point.clone().addScaledVector(perpendicular, -width / 2);
      positions.push(left.x, left.y, left.z, right.x, right.y, right.z);
      uvs.push(0, t, 1, t);
      if (index < segments) {
        const row = index * 2;
        indices.push(row, row + 2, row + 1, row + 1, row + 2, row + 3);
      }
    }
    const result = new BufferGeometry();
    result.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(positions), 3),
    );
    result.setAttribute("uv", new BufferAttribute(new Float32Array(uvs), 2));
    result.setIndex(indices);
    result.computeVertexNormals();
    return result;
  }, [points, width]);
  useEffect(
    () => () => {
      geometry.dispose();
    },
    [geometry],
  );

  return (
    <mesh
      geometry={geometry}
      renderOrder={layer}
      onClick={onClick}
      onPointerOver={
        onClick
          ? () => {
              document.body.style.cursor = "pointer";
            }
          : undefined
      }
      onPointerOut={
        onClick
          ? () => {
              document.body.style.cursor = "";
            }
          : undefined
      }
    >
      <meshBasicMaterial
        color={color}
        transparent={opacity < 1}
        opacity={opacity}
        side={DoubleSide}
        depthWrite={layer <= 2 && opacity === 1}
        polygonOffset={layer > 0}
        polygonOffsetFactor={-layer}
        polygonOffsetUnits={-layer}
        toneMapped={false}
      />
    </mesh>
  );
}

type RibbonRoute = {
  points: Vec3[];
  width: number;
};

function RibbonNetwork({
  routes,
  color,
  height = 0,
}: {
  routes: RibbonRoute[];
  color: string;
  height?: number;
}) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    const indices: number[] = [];
    let vertexOffset = 0;
    routes.forEach((route) => {
      const curve = new CatmullRomCurve3(
        route.points.map(([x, y, z]) => new Vector3(x, y + height, z)),
        false,
        "catmullrom",
        0.35,
      );
      const segments = Math.max(
        4,
        Math.min(18, (route.points.length - 1) * 5),
      );
      for (let index = 0; index <= segments; index += 1) {
        const progress = index / segments;
        const point = curve.getPoint(progress);
        const tangent = curve.getTangent(progress);
        const perpendicular = new Vector3(
          -tangent.z,
          0,
          tangent.x,
        ).normalize();
        const left = point
          .clone()
          .addScaledVector(perpendicular, route.width / 2);
        const right = point
          .clone()
          .addScaledVector(perpendicular, -route.width / 2);
        positions.push(left.x, left.y, left.z, right.x, right.y, right.z);
        if (index < segments) {
          const row = vertexOffset + index * 2;
          indices.push(row, row + 2, row + 1, row + 1, row + 2, row + 3);
        }
      }
      vertexOffset += (segments + 1) * 2;
    });
    const result = new BufferGeometry();
    result.setAttribute(
      "position",
      new BufferAttribute(new Float32Array(positions), 3),
    );
    result.setIndex(indices);
    return result;
  }, [routes, height]);
  useEffect(
    () => () => {
      geometry.dispose();
    },
    [geometry],
  );

  return (
    <mesh geometry={geometry} renderOrder={2}>
      <meshBasicMaterial
        color={color}
        side={DoubleSide}
        polygonOffset
        polygonOffsetFactor={-2}
        polygonOffsetUnits={-2}
        toneMapped={false}
      />
    </mesh>
  );
}

function GroundRoad({
  points,
  width,
  selected = false,
  onClick,
}: {
  points: Vec3[];
  width: number;
  selected?: boolean;
  onClick?: (event: ThreeEvent<MouseEvent>) => void;
}) {
  const raised = (offset: number) =>
    points.map(([x, y, z]) => [x, y + offset, z] as Vec3);
  return (
    <>
      <SurfaceRibbon
        points={raised(0)}
        color="#879388"
        width={width + 0.58}
        opacity={0.98}
        layer={1}
      />
      <SurfaceRibbon
        points={raised(0.018)}
        color={selected ? "#7b6939" : "#35413f"}
        width={width}
        opacity={1}
        layer={2}
        onClick={onClick}
      />
    </>
  );
}

function BuildingSignals({
  world,
  historyFiles,
}: {
  world: World;
  historyFiles: Map<string, number>;
}) {
  const { positions, colors } = useMemo(() => {
    const pointPositions: number[] = [];
    const pointColors: number[] = [];
    const testColor = new Color("#fff2b8");
    const activityColor = new Color("#35d9ff");
    world.buildings.forEach((building) => {
      const historicBytes = historyFiles.get(building.id);
      if (historicBytes === undefined) return;
      const ratio = Math.sqrt(
        Math.max(
          0.12,
          Math.min(3.2, historicBytes / Math.max(1, building.bytes)),
        ),
      );
      const height = building.scale[1] * ratio;
      const testLights = Math.min(6, Math.ceil(Math.log2(building.tests + 1)));
      const activityLights = Math.min(4, building.activity);
      const lights = testLights + activityLights;
      for (let index = 0; index < lights; index += 1) {
        const side = index % 2 === 0 ? 1 : -1;
        const offset = building.scale[0] * ratio * 0.53 * side;
        pointPositions.push(
          building.position[0] + Math.cos(building.rotation) * offset,
          0.5 + ((index + 1) / (lights + 1)) * Math.max(0.4, height - 0.5),
          building.position[2] - Math.sin(building.rotation) * offset,
        );
        (index < testLights ? testColor : activityColor).toArray(
          pointColors,
          pointColors.length,
        );
      }
    });
    return {
      positions: new Float32Array(pointPositions),
      colors: new Float32Array(pointColors),
    };
  }, [world, historyFiles]);

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        vertexColors
        size={0.16}
        sizeAttenuation
        transparent
        opacity={0.9}
        depthWrite={false}
      />
    </points>
  );
}

function UrbanGeography({
  data,
  world,
  historyFiles,
  historyAge,
}: {
  data: CityData;
  world: World;
  historyFiles: Map<string, number>;
  historyAge: number;
}) {
  const history = [...data.history].reverse();
  const minBytes = Math.min(...history.map((snapshot) => snapshot.bytes));
  const maxBytes = Math.max(...history.map((snapshot) => snapshot.bytes));
  const river = history.map((snapshot, index) => {
    const progress = index / Math.max(1, history.length - 1);
    const mass =
      (snapshot.bytes - minBytes) / Math.max(1, maxBytes - minBytes);
    return [
      -210 + progress * 420,
      0.035,
      -52 + mass * 44 + Math.sin(progress * Math.PI * 3) * 9,
    ] as Vec3;
  });
  return (
    <>
      {river.slice(0, -1).map((point, index) => {
        const snapshot = history[index + 1];
        const mass =
          (snapshot.bytes - minBytes) / Math.max(1, maxBytes - minBytes);
        return (
          <SurfaceRibbon
            key={`history-river-${snapshot.sha}`}
            points={[point, river[index + 1]]}
            color="#3f94ac"
            width={4 + mass * 8}
            opacity={0.86}
          />
        );
      })}
      <SurfaceRibbon
        points={river.map(([x, y, z]) => [x, y + 0.015, z] as Vec3)}
        color="#acdce2"
        width={0.38}
        opacity={0.72}
      />
      {history.map((snapshot, index) => {
        const point = river[index];
        const mass =
          (snapshot.bytes - minBytes) / Math.max(1, maxBytes - minBytes);
        return (
          <mesh
            key={`history-marker-${snapshot.sha}`}
            position={[point[0], 0.1, point[2]]}
            rotation={[-Math.PI / 2, 0, 0]}
            scale={[0.8 + mass * 1.4, 0.8 + mass * 1.4, 1]}
          >
            <circleGeometry args={[1, 20]} />
            <meshBasicMaterial color="#e4fbff" side={DoubleSide} />
          </mesh>
        );
      })}
      {world.districts.flatMap((district) => {
        const active = world.buildings.filter(
          (building) =>
            building.district === district.id && historyFiles.has(building.id),
        );
        const neighborhoods = new Map<string, BuildingWorld[]>();
        active.forEach((building) => {
          const rows = neighborhoods.get(building.neighborhood) ?? [];
          rows.push(building);
          neighborhoods.set(building.neighborhood, rows);
        });
        const neighborhoodRows = [...neighborhoods]
          .map(([neighborhood, buildings]) => ({
            neighborhood,
            buildings,
            x:
              buildings.reduce(
                (sum, building) => sum + building.position[0],
                0,
              ) / buildings.length,
            z:
              buildings.reduce(
                (sum, building) => sum + building.position[2],
                0,
              ) / buildings.length,
            mass: buildings.reduce((sum, building) => sum + building.loc, 0),
          }))
          .sort((a, b) => b.mass - a.mass);
        const connected: { x: number; z: number }[] = [
          { x: district.position[0], z: district.position[2] },
        ];
        const streetRoutes = neighborhoodRows.flatMap((neighborhood) => {
          const parent = connected.reduce((nearest, candidate) => {
            const nearestDistance = Math.hypot(
              neighborhood.x - nearest.x,
              neighborhood.z - nearest.z,
            );
            const candidateDistance = Math.hypot(
              neighborhood.x - candidate.x,
              neighborhood.z - candidate.z,
            );
            return candidateDistance < nearestDistance ? candidate : nearest;
          }, connected[0]);
          connected.push({ x: neighborhood.x, z: neighborhood.z });
          const arterialWidth = Math.min(
            1.28,
            0.24 +
              Math.log2(neighborhood.buildings.length + 1) * 0.11 +
              Math.log10(neighborhood.mass + 10) * 0.04,
          );
          const ordered = [...neighborhood.buildings].sort(
            (a, b) =>
              Math.atan2(
                a.position[2] - neighborhood.z,
                a.position[0] - neighborhood.x,
              ) -
              Math.atan2(
                b.position[2] - neighborhood.z,
                b.position[0] - neighborhood.x,
              ),
          );
          const stride = Math.max(1, Math.ceil(ordered.length / 12));
          const accessPoints = ordered
            .filter((_, index) => index % stride === 0)
            .map((building) => {
              const dx = building.position[0] - neighborhood.x;
              const dz = building.position[2] - neighborhood.z;
              const length = Math.max(0.1, Math.hypot(dx, dz));
              const clearance =
                Math.max(building.scale[0], building.scale[2]) * 0.72 + 0.22;
              return [
                building.position[0] + (dx / length) * clearance,
                0.03,
                building.position[2] + (dz / length) * clearance,
              ] as Vec3;
            });
          if (accessPoints.length > 2) accessPoints.push(accessPoints[0]);
          return [
            {
              points: [
                [parent.x, 0.03, parent.z],
                [neighborhood.x, 0.03, neighborhood.z],
              ] as Vec3[],
              width: arterialWidth + 0.16,
            },
            ...(accessPoints.length > 1
              ? [
                  {
                    points: accessPoints,
                    width:
                      Math.min(
                  0.46,
                  0.13 +
                    Math.log2(neighborhood.buildings.length + 1) * 0.045,
                      ) + 0.16,
                  },
                ]
              : []),
          ];
        });
        const completedIssues = new Set<number>();
        active.forEach((building) => {
          building.issueRefs.forEach((number) => {
            const issue = world.issueByNumber.get(number);
            if (
              issue?.closedAgeDays !== null &&
              issue?.closedAgeDays !== undefined &&
              issue.closedAgeDays >= historyAge
            ) {
              completedIssues.add(number);
            }
          });
        });
        const parkRadius =
          0.7 + Math.log2(completedIssues.size + 1) * 0.52;
        return [
          <RibbonNetwork
            key={`${district.id}-street-network`}
            routes={streetRoutes}
            color="#59635d"
            height={0.022}
          />,
          <mesh
            key={`${district.id}-park`}
            position={[district.position[0], 0.014, district.position[2]]}
            rotation={[-Math.PI / 2, 0, 0]}
            scale={[parkRadius, parkRadius * 0.78, 1]}
          >
            <circleGeometry args={[1, 24]} />
            <meshStandardMaterial
              color="#74a86f"
              roughness={1}
              side={DoubleSide}
            />
          </mesh>,
        ];
      })}
    </>
  );
}

function IssueSites({
  world,
  onSelect,
  historyAge,
}: {
  world: World;
  onSelect: (selection: Selection) => void;
  historyAge: number;
}) {
  const visibleIssues = useMemo(
    () =>
      world.issues
        .filter((issue) => issue.createdAgeDays >= historyAge)
        .map((issue) => {
          const wasStillOpen =
            issue.state === "open" ||
            (issue.closedAgeDays !== null && issue.closedAgeDays < historyAge);
          const elapsedAge = Math.max(0, issue.createdAgeDays - historyAge);
          const historicComments = Math.round(
            issue.comments *
              Math.min(1, elapsedAge / Math.max(1, issue.createdAgeDays)),
          );
          return {
            ...issue,
            state: wasStillOpen ? "open" : "closed",
            height: wasStillOpen
              ? 2.4 +
                Math.log2(historicComments + 1) * 1.5 +
                Math.min(3.2, elapsedAge / 420)
              : 0.18,
          };
        }),
    [world, historyAge],
  );
  const open = useMemo(
    () => visibleIssues.filter((issue) => issue.state === "open"),
    [visibleIssues],
  );
  const closed = useMemo(
    () => visibleIssues.filter((issue) => issue.state === "closed"),
    [visibleIssues],
  );
  const openRef = useRef<InstancedMesh>(null);
  const closedRef = useRef<InstancedMesh>(null);

  useLayoutEffect(() => {
    if (openRef.current) {
      open.forEach((issue, index) => {
        tempObject.position.set(
          issue.position[0],
          issue.height / 2 + 0.08,
          issue.position[2],
        );
        tempObject.rotation.set(0, hash(issue.number) * Math.PI, 0);
        tempObject.scale.set(issue.radius, issue.height, issue.radius);
        tempObject.updateMatrix();
        openRef.current?.setMatrixAt(index, tempObject.matrix);
      });
      openRef.current.instanceMatrix.needsUpdate = true;
    }
    if (closedRef.current) {
      closed.forEach((issue, index) => {
        tempObject.position.set(issue.position[0], 0.16, issue.position[2]);
        tempObject.scale.set(issue.radius, 0.2, issue.radius);
        tempObject.updateMatrix();
        closedRef.current?.setMatrixAt(index, tempObject.matrix);
      });
      closedRef.current.instanceMatrix.needsUpdate = true;
    }
  }, [open, closed]);

  const selectIssue =
    (rows: IssueWorld[]) => (event: ThreeEvent<MouseEvent>) => {
      event.stopPropagation();
      if (event.instanceId === undefined) return;
      onSelect({ kind: "issue", id: rows[event.instanceId].number });
    };

  return (
    <>
      <instancedMesh
        ref={openRef}
        args={[new CylinderGeometry(1, 0.72, 1, 8, 4), undefined, open.length]}
        onClick={selectIssue(open)}
        onPointerOver={() => {
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "";
        }}
      >
        <meshBasicMaterial
          color="#e7783d"
          transparent
          opacity={0.88}
          toneMapped={false}
        />
      </instancedMesh>
      <instancedMesh
        ref={closedRef}
        args={[new CylinderGeometry(1, 1, 1, 12), undefined, closed.length]}
        onClick={selectIssue(closed)}
        onPointerOver={() => {
          document.body.style.cursor = "pointer";
        }}
        onPointerOut={() => {
          document.body.style.cursor = "";
        }}
      >
        <meshBasicMaterial color="#45b77f" toneMapped={false} />
      </instancedMesh>
    </>
  );
}

function IssueConnections({
  data,
  world,
  historyAge,
}: {
  data: CityData;
  world: World;
  historyAge: number;
}) {
  const geometry = useMemo(() => {
    const positions: number[] = [];
    for (const edge of data.issueEdges) {
      const source = world.issueByNumber.get(edge.source);
      const target = world.issueByNumber.get(edge.target);
      if (
        !source ||
        !target ||
        source.createdAgeDays < historyAge ||
        target.createdAgeDays < historyAge
      ) {
        continue;
      }
      positions.push(
        source.position[0],
        0.45,
        source.position[2],
        target.position[0],
        0.45,
        target.position[2],
      );
    }
    const result = new BufferGeometry();
    result.setAttribute("position", new BufferAttribute(new Float32Array(positions), 3));
    return result;
  }, [data, world, historyAge]);
  return (
    <lineSegments geometry={geometry}>
      <lineBasicMaterial
        color="#b58cff"
        transparent
        opacity={0.095}
        depthWrite={false}
      />
    </lineSegments>
  );
}

function connectionRoutePoints(
  edge: DistrictConnection,
  source: DistrictWorld,
  target: DistrictWorld,
) {
  const corridorDx = target.position[0] - source.position[0];
  const corridorDz = target.position[2] - source.position[2];
  const corridorLength = Math.max(1, Math.hypot(corridorDx, corridorDz));
  const directionX = corridorDx / corridorLength;
  const directionZ = corridorDz / corridorLength;
  const perpendicularX = -corridorDz / corridorLength;
  const perpendicularZ = corridorDx / corridorLength;
  const sourceExitDistance =
    Math.min(source.width, source.depth) * 0.36;
  const targetExitDistance =
    Math.min(target.width, target.depth) * 0.36;
  const sourceExit: Vec3 = [
    source.position[0] + directionX * sourceExitDistance,
    0.034,
    source.position[2] + directionZ * sourceExitDistance,
  ];
  const targetExit: Vec3 = [
    target.position[0] - directionX * targetExitDistance,
    0.034,
    target.position[2] - directionZ * targetExitDistance,
  ];
  const relationBalance =
    (edge.dependency - edge.sharedIssues) / Math.max(1, edge.strength);
  const collapseOffset = MathUtils.clamp(
    relationBalance * corridorLength * 0.11,
    -8,
    8,
  );
  const roadHeight = edge.level === 0 ? 0.034 : 0.34 + edge.level * 0.72;
  const interpolate = (progress: number, lateral = 0): Vec3 => [
    MathUtils.lerp(sourceExit[0], targetExit[0], progress) +
      perpendicularX * lateral,
    progress === 0 || progress === 1 ? 0.034 : roadHeight,
    MathUtils.lerp(sourceExit[2], targetExit[2], progress) +
      perpendicularZ * lateral,
  ];
  return [
    interpolate(0),
    interpolate(0.2, collapseOffset * 0.42),
    interpolate(0.5, collapseOffset),
    interpolate(0.8, collapseOffset * 0.42),
    interpolate(1),
  ];
}

function DependencyRoads({
  world,
  selection,
  onSelect,
}: {
  world: World;
  selection: Selection;
  onSelect: (selection: Selection) => void;
}) {
  return (
    <>
      {world.connections.slice(0, 72).map((edge) => {
        const source = world.districtById.get(edge.source);
        const target = world.districtById.get(edge.target);
        if (!source || !target) return null;
        const corridorDx = target.position[0] - source.position[0];
        const corridorDz = target.position[2] - source.position[2];
        const corridorLength = Math.max(1, Math.hypot(corridorDx, corridorDz));
        const perpendicularX = -corridorDz / corridorLength;
        const perpendicularZ = corridorDx / corridorLength;
        const points = connectionRoutePoints(edge, source, target);
        const lanes = [
          {
            key: "dependency",
            value: edge.dependency,
            color: "#2876df",
          },
          {
            key: "co-change",
            value: edge.coChange,
            color: "#ffffff",
          },
          {
            key: "shared-issues",
            value: edge.sharedIssues,
            color: "#a85fdb",
          },
        ].filter((lane) => lane.value > 0);
        const selected =
          selection?.kind === "connection" && selection.id === edge.id;
        const roadWidth = Math.min(
          4.4,
          0.58 +
            Math.log2(edge.strength + 1) * 0.24 +
            edge.betweenness * 1.25,
        );
        const laneWidth = Math.max(0.16, roadWidth / (lanes.length + 1) * 0.58);
        return (
          <group key={edge.id}>
            {edge.level > 0
              ? [points[1], points[3]].map((point, supportIndex) => (
                  <mesh
                    key={`${edge.id}-support-${supportIndex}`}
                    position={[point[0], point[1] / 2, point[2]]}
                    scale={[0.32 + roadWidth * 0.08, point[1], 0.32 + roadWidth * 0.08]}
                  >
                    <cylinderGeometry args={[1, 1.15, 1, 8]} />
                    <meshBasicMaterial color="#68736e" toneMapped={false} />
                  </mesh>
                ))
              : null}
            <GroundRoad
              points={points}
              width={roadWidth}
              selected={selected}
              onClick={(event) => {
                event.stopPropagation();
                onSelect({ kind: "connection", id: edge.id });
              }}
            />
            {lanes.map((lane, laneIndex) => (
              <SurfaceRibbon
                key={`${edge.id}-${lane.key}`}
                points={points.map(([x, y, z]) => [
                  x +
                    perpendicularX *
                      (laneIndex - (lanes.length - 1) / 2) *
                      laneWidth *
                      1.35,
                  y + 0.058,
                  z +
                    perpendicularZ *
                      (laneIndex - (lanes.length - 1) / 2) *
                      laneWidth *
                      1.35,
                ])}
                color={lane.color}
                opacity={0.92}
                width={laneWidth}
                layer={4}
                onClick={(event) => {
                  event.stopPropagation();
                  onSelect({ kind: "connection", id: edge.id });
                }}
              />
            ))}
          </group>
        );
      })}
    </>
  );
}

type RoadVehicle = {
  curve: CatmullRomCurve3;
  phase: number;
  speed: number;
};

function TrafficStream({
  vehicles,
  color,
  scale,
  captureMode,
}: {
  vehicles: RoadVehicle[];
  color: string;
  scale: Vec3;
  captureMode: boolean;
}) {
  const meshRef = useRef<InstancedMesh>(null);
  const geometry = useMemo(() => new BoxGeometry(1, 1, 1), []);
  const vehicleObject = useMemo(() => new Object3D(), []);

  useFrame(({ clock }) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    vehicles.forEach((vehicle, index) => {
      const elapsed = captureMode ? 6 : clock.elapsedTime;
      const progress = (vehicle.phase + elapsed * vehicle.speed) % 1;
      const point = vehicle.curve.getPointAt(progress);
      const tangent = vehicle.curve.getTangentAt(progress);
      vehicleObject.position.set(point.x, point.y + 0.15, point.z);
      vehicleObject.rotation.set(
        0,
        Math.atan2(tangent.x, tangent.z),
        0,
      );
      vehicleObject.scale.set(scale[0], scale[1], scale[2]);
      vehicleObject.updateMatrix();
      mesh.setMatrixAt(index, vehicleObject.matrix);
    });
    mesh.instanceMatrix.needsUpdate = true;
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, undefined, vehicles.length]}
      frustumCulled={false}
    >
      <meshBasicMaterial color={color} toneMapped={false} />
    </instancedMesh>
  );
}

function RoadTraffic({
  data,
  world,
  days,
  historyAge,
  captureMode,
}: {
  data: CityData;
  world: World;
  days: number;
  historyAge: number;
  captureMode: boolean;
}) {
  const streams = useMemo(() => {
    const dependency: RoadVehicle[] = [];
    const coChange: RoadVehicle[] = [];
    const sharedIssues: RoadVehicle[] = [];
    const visibleCommits = new Set(
      data.commits
        .filter((commit) => {
          const age = ageDays(commit.date, data.generatedAt);
          return age >= historyAge && age <= historyAge + days;
        })
        .map((commit) => commit.sha),
    );
    const fileDistrict = new Map(
      data.files.map((file) => [file.id, file.district]),
    );
    const commitDistricts = new Map<string, Set<string>>();
    data.commitFileEdges.forEach((edge) => {
      if (!visibleCommits.has(edge.commit)) return;
      const district = fileDistrict.get(edge.file);
      if (!district) return;
      const districts = commitDistricts.get(edge.commit) ?? new Set<string>();
      districts.add(district);
      commitDistricts.set(edge.commit, districts);
    });
    const recentPairs = new Map<string, number>();
    commitDistricts.forEach((districts) => {
      const values = [...districts].sort();
      values.forEach((source, index) => {
        values.slice(index + 1).forEach((target) => {
          const key = `${source}::${target}`;
          recentPairs.set(key, (recentPairs.get(key) ?? 0) + 1);
        });
      });
    });
    world.connections.slice(0, 48).forEach((edge) => {
      const source = world.districtById.get(edge.source);
      const target = world.districtById.get(edge.target);
      if (!source || !target) return;
      const curve = new CatmullRomCurve3(
        connectionRoutePoints(edge, source, target).map(
          ([x, y, z]) => new Vector3(x, y + 0.078, z),
        ),
        false,
        "catmullrom",
        0.35,
      );
      const add = (
        rows: RoadVehicle[],
        kind: string,
        count: number,
        rate: number,
      ) => {
        for (let index = 0; index < count; index += 1) {
          rows.push({
            curve,
            phase: hash(`${edge.id}-${kind}-${index}`),
            speed: rate + Math.log2(edge.strength + 1) * 0.0012,
          });
        }
      };
      add(
        dependency,
        "dependency",
        Math.min(2, edge.dependency),
        0.018,
      );
      add(
        coChange,
        "co-change",
        Math.min(4, Math.ceil(Math.log2((recentPairs.get(edge.id) ?? 0) + 1))),
        0.026,
      );
      add(
        sharedIssues,
        "shared-issues",
        Math.min(2, Math.ceil(Math.log2(edge.sharedIssues + 1) / 2)),
        0.021,
      );
    });
    return { dependency, coChange, sharedIssues };
  }, [data, world, days, historyAge]);

  return (
    <>
      <TrafficStream
        vehicles={streams.dependency}
        color="#4f8ff7"
        scale={[0.48, 0.18, 0.24]}
        captureMode={captureMode}
      />
      <TrafficStream
        vehicles={streams.coChange}
        color="#fff9dc"
        scale={[0.34, 0.14, 0.2]}
        captureMode={captureMode}
      />
      <TrafficStream
        vehicles={streams.sharedIssues}
        color="#c081ee"
        scale={[0.4, 0.16, 0.22]}
        captureMode={captureMode}
      />
    </>
  );
}

function FailureFire({
  data,
  world,
  onSelect,
  captureMode,
}: {
  data: CityData;
  world: World;
  onSelect: (selection: Selection) => void;
  captureMode: boolean;
}) {
  const flameRef = useRef<InstancedMesh>(null);
  const coreRef = useRef<InstancedMesh>(null);
  const smokeRef = useRef<Points>(null);
  const flameGeometry = useMemo(() => new ConeGeometry(1, 1, 9), []);
  const incidents = useMemo(() => {
    const grouped = new Map<
      string,
      { building: BuildingWorld; count: number; severe: number }
    >();
    data.failures.forEach((failure) => {
      const path = canonicalFailureFile(failure.file);
      const building = path ? world.buildingById.get(path) : undefined;
      if (!building) return;
      const incident = grouped.get(building.id) ?? {
        building,
        count: 0,
        severe: 0,
      };
      incident.count += 1;
      if (failure.status === "TIMEOUT" || failure.status === "TERMINATING") {
        incident.severe += 1;
      }
      grouped.set(building.id, incident);
    });
    return [...grouped.values()].sort(
      (a, b) => b.count + b.severe - (a.count + a.severe),
    );
  }, [data, world]);
  const smokePositions = useMemo(
    () => new Float32Array(incidents.length * 3),
    [incidents],
  );
  const glowPositions = useMemo(() => {
    const positions = new Float32Array(incidents.length * 3);
    incidents.forEach((incident, index) => {
      const top =
        incident.building.position[1] + incident.building.scale[1] / 2;
      positions[index * 3] = incident.building.position[0];
      positions[index * 3 + 1] = top + 1.8;
      positions[index * 3 + 2] = incident.building.position[2];
    });
    return positions;
  }, [incidents]);

  useEffect(() => () => flameGeometry.dispose(), [flameGeometry]);

  useFrame(({ clock }) => {
    const t = captureMode ? 6 : clock.elapsedTime;
    incidents.forEach((incident, index) => {
      const phase = hash(incident.building.id);
      const top =
        incident.building.position[1] + incident.building.scale[1] / 2;
      const height =
        2.35 +
        Math.log2(incident.count + 1) * 0.68 +
        Math.log2(incident.severe + 1) * 0.58;
      const width = 0.68 + Math.log2(incident.count + 1) * 0.14;
      const breath = 1 + Math.sin(t * 4.2 + phase * Math.PI * 2) * 0.09;
      const lean = Math.sin(t * 2.1 + phase * 11) * width * 0.18;

      tempObject.position.set(
        incident.building.position[0] + lean,
        top + (height * breath) / 2,
        incident.building.position[2],
      );
      tempObject.rotation.set(0, phase * Math.PI, -lean * 0.08);
      tempObject.scale.set(width / breath, height * breath, width / breath);
      tempObject.updateMatrix();
      flameRef.current?.setMatrixAt(index, tempObject.matrix);

      tempObject.position.set(
        incident.building.position[0] - lean * 0.25,
        top + height * 0.3,
        incident.building.position[2],
      );
      tempObject.rotation.set(0, phase * Math.PI * 1.7, lean * 0.05);
      tempObject.scale.set(width * 0.48, height * 0.58 * breath, width * 0.48);
      tempObject.updateMatrix();
      coreRef.current?.setMatrixAt(index, tempObject.matrix);

      const cycle = (phase + t * (0.08 + phase * 0.05)) % 1;
      smokePositions[index * 3] =
        incident.building.position[0] + Math.sin(t + index) * cycle * 0.45;
      smokePositions[index * 3 + 1] = top + height * 0.72 + cycle * 5.2;
      smokePositions[index * 3 + 2] =
        incident.building.position[2] + Math.cos(t * 0.8 + index) * cycle * 0.45;
    });
    if (flameRef.current) flameRef.current.instanceMatrix.needsUpdate = true;
    if (coreRef.current) coreRef.current.instanceMatrix.needsUpdate = true;
    const attribute = smokeRef.current?.geometry.getAttribute("position");
    if (attribute) attribute.needsUpdate = true;
  });

  return (
    <>
      <instancedMesh
        ref={flameRef}
        args={[flameGeometry, undefined, incidents.length]}
        frustumCulled={false}
        onClick={(event: ThreeEvent<MouseEvent>) => {
          event.stopPropagation();
          if (event.instanceId === undefined) return;
          const incident = incidents[event.instanceId];
          onSelect({ kind: "building", id: incident.building.id });
        }}
      >
        <meshBasicMaterial
          color="#ff672d"
          transparent
          opacity={0.96}
          toneMapped={false}
        />
      </instancedMesh>
      <instancedMesh
        ref={coreRef}
        args={[flameGeometry, undefined, incidents.length]}
        frustumCulled={false}
      >
        <meshBasicMaterial
          color="#ffe36f"
          transparent
          opacity={0.98}
          toneMapped={false}
          depthWrite={false}
        />
      </instancedMesh>
      <points frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[glowPositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color="#ff8a36"
          size={1.9}
          sizeAttenuation
          transparent
          opacity={0.58}
          depthWrite={false}
          blending={AdditiveBlending}
          toneMapped={false}
        />
      </points>
      <points ref={smokeRef} frustumCulled={false}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[smokePositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          color="#59676a"
          size={0.86}
          sizeAttenuation
          transparent
          opacity={0.46}
          depthWrite={false}
        />
      </points>
    </>
  );
}

function CommentCrowds({
  world,
  historyAge,
  captureMode,
}: {
  world: World;
  historyAge: number;
  captureMode: boolean;
}) {
  const pointsRef = useRef<Points>(null);
  const crowd = useMemo(() => {
    const actors: {
      x: number;
      z: number;
      angle: number;
      radius: number;
      y: number;
      pace: number;
      phase: number;
    }[] = [];
    for (const issue of world.issues) {
      if (issue.createdAgeDays < historyAge) continue;
      const lifeElapsed = Math.max(0, issue.createdAgeDays - historyAge);
      const elapsedFraction = Math.min(
        1,
        lifeElapsed / Math.max(1, issue.createdAgeDays),
      );
      const historicalComments = Math.round(issue.comments * elapsedFraction);
      const count = Math.min(
        16,
        Math.ceil(Math.log2(historicalComments + 1) * 2),
      );
      for (let index = 0; index < count; index += 1) {
        const angle = hash(`${issue.number}-${index}`) * Math.PI * 2;
        const radius = issue.radius + 0.8 + (index % 4) * 0.22;
        const phase = hash(`crowd-${issue.number}-${index}`);
        actors.push({
          x: issue.position[0],
          z: issue.position[2],
          angle,
          radius,
          y: 0.25 + (index % 3) * 0.12,
          pace: 0.045 + phase * 0.055,
          phase,
        });
      }
    }
    return {
      actors,
      positions: new Float32Array(actors.length * 3),
    };
  }, [world, historyAge]);

  useFrame(({ clock }) => {
    const t = captureMode ? 6 : clock.elapsedTime;
    crowd.actors.forEach((actor, index) => {
      const angle = actor.angle + t * actor.pace;
      crowd.positions[index * 3] = actor.x + Math.cos(angle) * actor.radius;
      crowd.positions[index * 3 + 1] =
        actor.y + Math.sin(t * 2.4 + actor.phase * 12) * 0.055;
      crowd.positions[index * 3 + 2] =
        actor.z + Math.sin(angle) * actor.radius;
    });
    const attribute = pointsRef.current?.geometry.getAttribute("position");
    if (attribute) attribute.needsUpdate = true;
  });

  return (
    <points ref={pointsRef} frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          args={[crowd.positions, 3]}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#ffc88c"
        size={0.3}
        transparent
        opacity={0.86}
        depthWrite={false}
      />
    </points>
  );
}

function RunWeather({
  runs,
  world,
  captureMode,
}: {
  runs: Run[];
  world: World;
  captureMode: boolean;
}) {
  const pointsRef = useRef<Points>(null);
  const tetherRef = useRef<BufferGeometry>(null);
  const positions = useMemo(() => new Float32Array(runs.length * 3), [runs]);
  const tetherPositions = useMemo(
    () => new Float32Array(runs.length * 6),
    [runs],
  );
  const workflowBuildings = useMemo(
    () => world.buildings.filter((building) => building.kind === "workflow"),
    [world],
  );
  const workflowCenter = useMemo(() => {
    if (workflowBuildings.length === 0) return null;
    return [
      workflowBuildings.reduce(
        (sum, building) => sum + building.position[0],
        0,
      ) / workflowBuildings.length,
      workflowBuildings.reduce(
        (sum, building) => sum + building.position[2],
        0,
      ) / workflowBuildings.length,
    ] as const;
  }, [workflowBuildings]);
  const colors = useMemo(() => {
    const result = new Float32Array(runs.length * 3);
    runs.forEach((run, index) => {
      let color = new Color("#8ca9b8");
      if (run.status === "in_progress") color = new Color("#5ee7ff");
      else if (run.status === "queued" || run.status === "pending") {
        color = new Color("#ffba6b");
      } else if (run.conclusion === "failure") color = new Color("#ff665e");
      else if (run.conclusion === "success") color = new Color("#6ce0a5");
      color.toArray(result, index * 3);
    });
    return result;
  }, [runs]);

  useFrame(({ clock }) => {
    const t = captureMode ? 6 : clock.elapsedTime;
    if (workflowBuildings.length === 0) return;
    runs.forEach((run, index) => {
      const anchor = workflowBuildings[run.id % workflowBuildings.length];
      const lane = index % 4;
      const angle = hash(run.id) * Math.PI * 2 + t * (0.015 + lane * 0.003);
      const radius = 2.4 + lane * 1.4 + (run.status === "queued" ? 1.8 : 0);
      positions[index * 3] = anchor.position[0] + Math.cos(angle) * radius;
      positions[index * 3 + 1] =
        7.5 + lane * 1.1 + Math.sin(t * 0.4 + index) * 0.7;
      positions[index * 3 + 2] =
        anchor.position[2] + Math.sin(angle) * radius;
      tetherPositions[index * 6] = anchor.position[0];
      tetherPositions[index * 6 + 1] =
        anchor.position[1] + anchor.scale[1] / 2;
      tetherPositions[index * 6 + 2] = anchor.position[2];
      tetherPositions[index * 6 + 3] = positions[index * 3];
      tetherPositions[index * 6 + 4] = positions[index * 3 + 1];
      tetherPositions[index * 6 + 5] = positions[index * 3 + 2];
    });
    const attribute = pointsRef.current?.geometry.getAttribute("position");
    if (attribute) attribute.needsUpdate = true;
    const tetherAttribute = tetherRef.current?.getAttribute("position");
    if (tetherAttribute) tetherAttribute.needsUpdate = true;
  });

  if (!workflowCenter) return null;

  return (
    <>
      <lineSegments>
        <bufferGeometry ref={tetherRef}>
          <bufferAttribute
            attach="attributes-position"
            args={[tetherPositions, 3]}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#dceaf0" transparent opacity={0.2} />
      </lineSegments>
      <points ref={pointsRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial
          vertexColors
          size={1}
          transparent
          opacity={0.85}
          depthWrite={false}
          blending={AdditiveBlending}
        />
      </points>
      <Html
        position={[workflowCenter[0], 13.5, workflowCenter[1]]}
        distanceFactor={34}
        zIndexRange={[7, 0]}
      >
        <div className="lag-label">
          <strong>GITHUB ACTIONS AIRSPACE</strong>
          <span>{runs.length} runs tethered to workflow files</span>
        </div>
      </Html>
    </>
  );
}

function measurementTargets(measurement: Measurement) {
  if (measurement.name === "rust-test-suite") {
    return ["gam-models", "gam-sae", "gam-solve", "gam-terms"];
  }
  if (measurement.name === "reference-quality") {
    return ["tests:quality"];
  }
  if (measurement.name === "python-contracts") {
    return ["python-api", "gam-pyffi"];
  }
  if (measurement.name === "fuzz") {
    return ["gam-solve", "measurement-labs"];
  }
  return ["measurement-labs"];
}

function MeasurementAtmosphere({
  data,
  world,
  historyAge,
}: {
  data: CityData;
  world: World;
  historyAge: number;
}) {
  if (historyAge > data.summary.measurementLagHours / 24 + 1) return null;

  return (
    <>
      {data.measurements.flatMap((measurement) =>
        measurementTargets(measurement).map((districtId, targetIndex) => {
          const district = world.districtById.get(districtId);
          if (!district) return null;
          const lagScale = Math.max(0.18, measurement.lagHours / 14);
          const warm =
            measurement.status === "failure" ||
            measurement.status === "cancelled";
          const drift = hash(`${measurement.name}-${districtId}`) * Math.PI * 2;
          return (
            <group
              key={`${measurement.name}-${districtId}`}
              position={[
                district.position[0] + Math.cos(drift) * district.width * 0.12,
                8 + lagScale * 5,
                district.position[2] + Math.sin(drift) * district.depth * 0.12,
              ]}
            >
              {[0, 1, 2].map((cloud) => (
                <mesh
                  key={cloud}
                  position={[
                    (cloud - 1) * 4.5,
                    Math.sin(cloud * 2.4) * 1.4,
                    (hash(`${measurement.name}-cloud-${cloud}`) - 0.5) * 5,
                  ]}
                  scale={[
                    (5.5 + cloud * 1.4) * lagScale,
                    (2.1 + cloud * 0.45) * lagScale,
                    (4.2 + cloud) * lagScale,
                  ]}
                >
                  <sphereGeometry args={[1, 20, 12]} />
                  <meshStandardMaterial
                    color={warm ? "#d5a39b" : "#bfd9d9"}
                    transparent
                    opacity={0.06 + lagScale * 0.07}
                    depthWrite={false}
                    roughness={1}
                  />
                </mesh>
              ))}
              {targetIndex === 0 ? (
                <Html position={[0, 5.3, 0]} distanceFactor={38} zIndexRange={[7, 0]}>
                  <div className="lag-label">
                    <strong>{measurement.name}</strong>
                    <span>{measurement.lagHours.toFixed(1)}h local visibility lag</span>
                  </div>
                </Html>
              ) : null}
            </group>
          );
        }),
      )}
    </>
  );
}

function Ground() {
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.08, 0]}>
      <planeGeometry args={[440, 350, 1, 1]} />
      <meshStandardMaterial color="#6e8c80" roughness={0.98} metalness={0.02} />
    </mesh>
  );
}

function CityScene({
  data,
  world,
  selection,
  onSelect,
  layers,
  days,
  view,
  historyFiles,
  historyAge,
  captureMode,
}: {
  data: CityData;
  world: World;
  selection: Selection;
  onSelect: (selection: Selection) => void;
  layers: {
    issues: boolean;
    failures: boolean;
    commits: boolean;
    dependencies: boolean;
    runs: boolean;
  };
  days: number;
  view: CameraView;
  historyFiles: Map<string, number>;
  historyAge: number;
  captureMode: boolean;
}) {
  return (
    <>
      <color attach="background" args={["#8dbbc7"]} />
      <fogExp2 attach="fog" args={["#9cc9d1", 0.00115]} />
      <ambientLight intensity={1.15} color="#d6f3f4" />
      <hemisphereLight args={["#f2fdff", "#4d6d62", 2.1]} />
      <directionalLight
        position={[55, 95, 35]}
        intensity={2.4}
        color="#fff4dc"
      />
      <Ground />
      <UrbanGeography
        data={data}
        world={world}
        historyFiles={historyFiles}
        historyAge={historyAge}
      />
      <Districts world={world} onSelect={onSelect} historyFiles={historyFiles} />
      {layers.dependencies ? (
        <>
          <DependencyRoads
            world={world}
            selection={selection}
            onSelect={onSelect}
          />
          {layers.commits ? (
            <RoadTraffic
              data={data}
              world={world}
              days={days}
              historyAge={historyAge}
              captureMode={captureMode}
            />
          ) : null}
        </>
      ) : null}
      <Buildings
        world={world}
        selection={selection}
        onSelect={onSelect}
        historyFiles={historyFiles}
      />
      <BuildingSignals world={world} historyFiles={historyFiles} />
      {layers.issues ? (
        <>
          <IssueConnections data={data} world={world} historyAge={historyAge} />
          <IssueSites world={world} onSelect={onSelect} historyAge={historyAge} />
          <CommentCrowds
            world={world}
            historyAge={historyAge}
            captureMode={captureMode}
          />
        </>
      ) : null}
      {layers.failures &&
      historyAge <= data.summary.measurementLagHours / 24 + 1 ? (
        <FailureFire
          data={data}
          world={world}
          onSelect={onSelect}
          captureMode={captureMode}
        />
      ) : null}
      {layers.runs ? (
        <RunWeather
          world={world}
          captureMode={captureMode}
          runs={data.runs.filter(
            (run) =>
              Math.abs(run.ageHours / 24 - historyAge) <= Math.max(1, days),
          )}
        />
      ) : null}
      <MeasurementAtmosphere data={data} world={world} historyAge={historyAge} />
      <CameraRig view={view} />
      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={10}
        maxDistance={210}
        maxPolarAngle={Math.PI / 2.03}
        target={[0, 3, 0]}
      />
    </>
  );
}

function SelectionPanel({
  selection,
  world,
  onClose,
}: {
  selection: Selection;
  world: World;
  onClose: () => void;
}) {
  if (!selection) return null;

  if (selection.kind === "building") {
    const building = world.buildingById.get(selection.id);
    if (!building) return null;
    return (
      <aside className="selection-panel" aria-label="Selected building">
        <header>
          <div>
            <span className="eyebrow">File building · {building.kind}</span>
            <h2>{building.path}</h2>
          </div>
          <button className="panel-close" onClick={onClose} aria-label="Close details">
            ×
          </button>
        </header>
        <div className="detail-grid">
          <Detail label="Physical mass" value={`${compact(building.loc)} LOC`} />
          <Detail label="District" value={building.district} />
          <Detail label="Test floors" value={String(building.tests)} />
          <Detail label="Active fires" value={String(building.failureCount)} />
          <Detail label="30d construction" value={`${building.activity} commits`} />
          <Detail
            label="Issue utilities"
            value={building.issueRefs.length ? building.issueRefs.map((n) => `#${n}`).join(" ") : "unlinked"}
          />
        </div>
        <p className="detail-note">
          Height is LOC. Footprint is file bytes plus issue and co-change coupling.
          Rooftop mass and façade lights are tests and recent construction. Red glass
          and rising embers are failing tests anchored to this address.
        </p>
      </aside>
    );
  }

  if (selection.kind === "issue") {
    const issue = world.issueByNumber.get(selection.id);
    if (!issue) return null;
    return (
      <aside className="selection-panel" aria-label="Selected issue">
        <header>
          <div>
            <span className="eyebrow">
              {issue.state === "open" ? "Construction site" : "Completed plaza"} · #
              {issue.number}
            </span>
            <h2>{issue.title}</h2>
          </div>
          <button className="panel-close" onClick={onClose} aria-label="Close details">
            ×
          </button>
        </header>
        <div className="detail-grid">
          <Detail label="State" value={issue.state} />
          <Detail label="Crowd mass" value={`${issue.comments} comments`} />
          <Detail label="Age" value={`${Math.round(issue.createdAgeDays)} days`} />
          <Detail label="Last movement" value={`${Math.round(issue.updatedAgeDays)}d ago`} />
          <Detail label="Cross-streets" value={`${issue.refs.length} issues`} />
          <Detail
            label="Labels"
            value={issue.labels.slice(0, 3).join(", ") || "none"}
          />
        </div>
        <p className="detail-note">
          Open work rises as scaffolding; age darkens it, comments gather a crowd,
          and links pull the site toward the files and issues it affects. Closing it
          replaces construction with a lit public plaza.
        </p>
        <a className="detail-link" href={issue.url} target="_blank" rel="noreferrer">
          Open issue on GitHub ↗
        </a>
      </aside>
    );
  }

  if (selection.kind === "connection") {
    const connection = world.connectionById.get(selection.id);
    if (!connection) return null;
    const layers = [
      connection.dependency > 0 ? "dependency" : null,
      connection.coChange > 0 ? "co-change" : null,
      connection.sharedIssues > 0 ? "shared issues" : null,
    ].filter(Boolean);
    return (
      <aside className="selection-panel" aria-label="Selected shared road">
        <header>
          <div>
            <span className="eyebrow">Shared inter-district road</span>
            <h2>
              {connection.source} ↔ {connection.target}
            </h2>
          </div>
          <button className="panel-close" onClick={onClose} aria-label="Close details">
            ×
          </button>
        </header>
        <div className="detail-grid">
          <Detail label="Cargo dependencies" value={String(connection.dependency)} />
          <Detail label="Co-change events" value={String(connection.coChange)} />
          <Detail label="Shared issue refs" value={String(connection.sharedIssues)} />
          <Detail label="Combined strength" value={connection.strength.toFixed(2)} />
          <Detail
            label="Network betweenness"
            value={`${(connection.betweenness * 100).toFixed(1)}%`}
          />
          <Detail
            label="Road grade"
            value={connection.level === 0 ? "ground" : `bridge ${connection.level}`}
          />
        </div>
        <p className="detail-note">
          One physical corridor carries every relationship these boroughs share.
          Blue is dependency traffic, white is files changed in the same commits,
          and violet is issue coupling. This road carries {layers.join(", ")}.
        </p>
      </aside>
    );
  }

  const district = world.districtById.get(selection.id);
  if (!district) return null;
  return (
    <aside className="selection-panel" aria-label="Selected district">
      <header>
        <div>
          <span className="eyebrow">Crate / system district</span>
          <h2>{district.id}</h2>
        </div>
        <button className="panel-close" onClick={onClose} aria-label="Close details">
          ×
        </button>
      </header>
      <div className="detail-grid">
        <Detail label="Buildings" value={compact(district.files)} />
        <Detail label="Urban mass" value={`${compact(district.loc)} LOC`} />
        <Detail label="Test capacity" value={compact(district.tests)} />
        <Detail label="Active fires" value={compact(district.failures)} />
        <Detail
          label="Graph centrality"
          value={`${(district.centrality * 100).toFixed(2)}%`}
        />
        <Detail label="Connectome community" value={String(district.community + 1)} />
      </div>
      <p className="detail-note">
        Weighted modularity forms the community, PageRank-like centrality sizes
        its influence, and the force embedding places it near the systems it
        actually exchanges dependencies, commits, and issues with.
      </p>
    </aside>
  );
}

function Detail({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-cell">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function CivicLaws({
  expanded,
  onToggle,
}: {
  expanded: boolean;
  onToggle: () => void;
}) {
  return (
    <div className="laws-wrap">
      {expanded ? (
        <section className="laws-panel" aria-label="How the city encodes the codebase">
          <h2>Civic laws — every event changes the city</h2>
          <div className="law-grid">
            <Law swatch="" title="Buildings" text="each file lives inside its real directory neighborhood; height = LOC, footprint = bytes plus coupling." />
            <Law swatch="" title="Façades" text="file kind selects material and structural texture; warm windows are tests, cyan windows are recent commits." />
            <Law swatch="road" title="Local streets" text="a minimum-distance directory tree joins neighborhoods; access loops follow their file buildings." />
            <Law swatch="road" title="Highways" text="edge betweenness and coupling determine road class; crossing analysis raises only the corridors that need bridges." />
            <Law swatch="road" title="Shared roads" text="one corridor carries blue dependencies, white co-changes, and violet shared issues; matching vehicles move on each lane." />
            <Law swatch="river" title="History river" text="each bend is a Git snapshot; widening water is the repository gaining source mass." />
            <Law swatch="success" title="Parks" text="district green space grows from issues completed in that part of the codebase." />
            <Law swatch="issue" title="Construction" text="open issues are warm faceted construction towers; age adds height and comments gather crowds." />
            <Law swatch="success" title="Completion" text="closed issues become low, permanent, illuminated public plazas." />
            <Law swatch="fire" title="Failure" text="a flame and smoke plume ignite the exact failing file; repeated failures and timeouts burn higher." />
            <Law swatch="" title="Commits" text="street-level moving light carries work from issue sites into changed buildings." />
            <Law swatch="fog" title="Measurement lag" text="each artifact creates its own local fog bank only over the systems it measures." />
            <Law swatch="issue" title="Actions weather" text="run markers hover only over workflow files, with a visible tether back to their source." />
            <Law swatch="" title="Connectome" text="weighted modularity finds communities; centrality and force embedding grow the organic metropolitan structure." />
          </div>
        </section>
      ) : null}
      <button
        className="laws-toggle"
        onClick={onToggle}
        aria-expanded={expanded}
      >
        {expanded ? "Hide civic laws" : "How this city behaves"}
      </button>
    </div>
  );
}

function NetworkKey() {
  return (
    <aside className="network-key" aria-label="Road and texture legend">
      <span>
        <i className="network-line dependency" /> dependency
      </span>
      <span>
        <i className="network-line cochange" /> co-change
      </span>
      <span>
        <i className="network-line issue-road" /> shared issues
      </span>
      <span>
        <i className="network-car" /> moving traffic
      </span>
      <span>
        <i className="network-dot tests" /> tests
      </span>
      <span>
        <i className="network-dot activity" /> recent commits
      </span>
      <span>
        <i className="network-flame" /> failing file
      </span>
      <span>
        <i className="network-tether" /> Actions run
      </span>
    </aside>
  );
}

function Law({
  swatch,
  title,
  text,
}: {
  swatch: string;
  title: string;
  text: string;
}) {
  return (
    <div className="law">
      <span className={`swatch ${swatch}`} aria-hidden="true" />
      <span>
        <b>{title}.</b> {text}
      </span>
    </div>
  );
}

function LoadedCodebaseCity({
  data,
  captureMode,
}: {
  data: CityData;
  captureMode: boolean;
}) {
  const world = useMemo(() => makeWorld(data), [data]);
  const [selection, setSelection] = useState<Selection>(null);
  const [days, setDays] = useState(30);
  const [view, setView] = useState<CameraView>("overview");
  const [laws, setLaws] = useState(!captureMode);
  const [timelinePosition, setTimelinePosition] = useState(
    data.history.length - 1,
  );
  const [playing, setPlaying] = useState(false);
  const [layers, setLayers] = useState({
    issues: true,
    failures: true,
    commits: true,
    dependencies: true,
    runs: true,
  });
  const activeRuns = data.runs.filter((run) =>
    ["in_progress", "queued", "pending"].includes(run.status),
  ).length;
  const historyIndex = data.history.length - 1 - timelinePosition;
  const snapshot = data.history[historyIndex];
  const historyFiles = useMemo(
    () => new Map(snapshot.files),
    [snapshot],
  );
  const historicOpenIssues = data.issues.filter((issue) => {
    if (issue.createdAgeDays < snapshot.ageDays) return false;
    return (
      issue.state === "open" ||
      (issue.closedAgeDays !== null && issue.closedAgeDays < snapshot.ageDays)
    );
  }).length;
  const snapshotDate = new Date(snapshot.date);

  useEffect(() => {
    if (!playing) return;
    if (timelinePosition >= data.history.length - 1) return;
    const timer = window.setTimeout(() => {
      setTimelinePosition((current) => current + 1);
      if (timelinePosition + 1 >= data.history.length - 1) {
        setPlaying(false);
      }
    }, 1250);
    return () => window.clearTimeout(timer);
  }, [data.history.length, playing, timelinePosition]);

  function toggleLayer(key: keyof typeof layers) {
    setLayers((current) => ({ ...current, [key]: !current[key] }));
  }

  return (
    <main
      className={`city-shell${captureMode ? " capture-mode" : ""}`}
      data-city-ready="true"
    >
      <div className="topbar">
        <div className="brand">
          <span className="eyebrow">SauersML / gam · live repository archaeology</span>
          <h1>Codebase City</h1>
          <p>
            {compact(snapshot.fileCount)} physical file-buildings across{" "}
            {data.districts.length} crate and system districts ·{" "}
            <span className="mono">{snapshot.label}</span> ·{" "}
            {snapshotTimestamp(snapshotDate)}.
          </p>
        </div>
        <div className="status-ribbon" aria-label="City status">
          <div className="stat-chip">
            <strong>{compact(snapshot.fileCount)}</strong>
            <span>visible buildings</span>
          </div>
          <div className="stat-chip">
            <strong>
              {historicOpenIssues} / {compact(data.summary.totalIssues)}
            </strong>
            <span>open / known issues</span>
          </div>
          <div className="stat-chip danger">
            <strong>
              {snapshot.ageDays <= data.summary.measurementLagHours / 24 + 1
                ? compact(data.summary.failures)
                : "unmeasured"}
            </strong>
            <span>test fires at time</span>
          </div>
          <div className="stat-chip">
            <strong>{compact(snapshot.bytes)}</strong>
            <span>source bytes</span>
          </div>
          <div className="stat-chip fog">
            <strong>{data.summary.measurementLagHours.toFixed(1)}h</strong>
            <span>max local lag</span>
          </div>
          <div className="stat-chip">
            <strong>{activeRuns}</strong>
            <span>active Actions tethers</span>
          </div>
        </div>
      </div>

      <NetworkKey />

      <SceneErrorBoundary>
        <Canvas
          className="city-canvas"
          dpr={[1, 1.35]}
          gl={{ antialias: true, alpha: false, powerPreference: "high-performance" }}
          camera={{ position: [90, 72, 96], fov: 46, near: 0.1, far: 400 }}
          onPointerMissed={() => setSelection(null)}
          aria-label="Interactive 3D city of the GAM repository"
        >
          <Suspense fallback={null}>
            <CityScene
              data={data}
              world={world}
              selection={selection}
              onSelect={setSelection}
              layers={layers}
              days={days}
              view={view}
              historyFiles={historyFiles}
              historyAge={snapshot.ageDays}
              captureMode={captureMode}
            />
          </Suspense>
        </Canvas>
      </SceneErrorBoundary>

      <SelectionPanel
        selection={selection}
        world={world}
        onClose={() => setSelection(null)}
      />

      {captureMode ? null : (
        <CivicLaws expanded={laws} onToggle={() => setLaws((value) => !value)} />
      )}

      {captureMode ? null : (
        <nav className="control-tray" aria-label="City controls">
        <div className="timeline-group">
          <button
            className="play-button"
            onClick={() => {
              if (timelinePosition >= data.history.length - 1) {
                setTimelinePosition(0);
              }
              setPlaying((current) => !current);
            }}
            aria-pressed={playing}
            aria-label={playing ? "Pause city evolution" : "Play city evolution"}
          >
            {playing ? "pause" : "play history"}
          </button>
          <label className="timeline-control">
            <span className="timeline-readout">
              <b>{snapshot.label}</b>
              <span>
                {snapshotDate.toISOString().slice(0, 10)} · {snapshot.sha.slice(0, 8)}
              </span>
            </span>
            <input
              className="timeline-range"
              type="range"
              min={0}
              max={data.history.length - 1}
              step={1}
              value={timelinePosition}
              aria-label="Repository history"
              onChange={(event) => {
                setPlaying(false);
                setTimelinePosition(Number(event.target.value));
              }}
            />
          </label>
        </div>
        <div className="control-group camera-controls">
          {(["overview", "street", "top"] as CameraView[]).map((cameraView) => (
            <button
              key={cameraView}
              className={view === cameraView ? "active" : ""}
              onClick={() => setView(cameraView)}
              aria-pressed={view === cameraView}
            >
              {cameraView}
            </button>
          ))}
        </div>
        <div className="control-group">
          <select
            aria-label="Commit history window"
            value={days}
            onChange={(event) => setDays(Number(event.target.value))}
          >
            <option value={1}>24h history</option>
            <option value={7}>7d history</option>
            <option value={30}>30d history</option>
            <option value={3650}>all history</option>
          </select>
        </div>
        <div className="control-group">
          {(Object.keys(layers) as (keyof typeof layers)[]).map((layer) => (
            <button
              key={layer}
              onClick={() => toggleLayer(layer)}
              aria-pressed={layers[layer]}
            >
              {layer === "dependencies" ? "roads" : layer}
            </button>
          ))}
        </div>
        </nav>
      )}
    </main>
  );
}

export function CodebaseCity() {
  const [data, setData] = useState<CityData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    fetch("/city-data.json")
      .then((response) => {
        if (!response.ok) {
          throw new Error(`City survey failed with HTTP ${response.status}`);
        }
        return response.json() as Promise<CityData>;
      })
      .then((snapshot) => {
        if (active) setData(snapshot);
      })
      .catch((reason: unknown) => {
        if (!active) return;
        setError(reason instanceof Error ? reason.message : "City survey failed");
      });
    return () => {
      active = false;
    };
  }, []);

  if (error) {
    return (
      <main className="city-shell">
        <div className="loading-fallback" role="alert">
          {error}
        </div>
      </main>
    );
  }
  if (!data) {
    return (
      <main className="city-shell">
        <div className="loading-fallback" role="status">
          Surveying repository geography…
        </div>
      </main>
    );
  }
  const captureMode =
    new URLSearchParams(window.location.search).get("capture") === "readme";
  return <LoadedCodebaseCity data={data} captureMode={captureMode} />;
}
