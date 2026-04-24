// Synthetic fixture that matches the expected Evidence Explorer
// shape: a forest of companies / people / transactions with
// ~3 edges per node on average. Deterministic (seeded) so every
// benchmark run compares to the same input.

export interface SpikeNode {
  id: string;
  label: string;
  type: "Company" | "Person" | "Transaction";
}

export interface SpikeEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface SpikeGraph {
  nodes: SpikeNode[];
  edges: SpikeEdge[];
}

// Tiny deterministic PRNG — we want fixture stability across runs
// without pulling in a dep.
function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function generateFixture(
  nodeCount: number,
  edgeMultiplier = 3,
  seed = 42,
): SpikeGraph {
  const rand = mulberry32(seed);
  const types: SpikeNode["type"][] = ["Company", "Person", "Transaction"];

  const nodes: SpikeNode[] = Array.from({ length: nodeCount }, (_, i) => ({
    id: `n${i}`,
    label: `${types[i % 3]}-${i}`,
    type: types[i % 3],
  }));

  const targetEdgeCount = nodeCount * edgeMultiplier;
  const edges: SpikeEdge[] = [];
  for (let i = 0; i < targetEdgeCount; i++) {
    const src = Math.floor(rand() * nodeCount);
    let tgt = Math.floor(rand() * nodeCount);
    if (tgt === src) tgt = (tgt + 1) % nodeCount;
    edges.push({
      id: `e${i}`,
      source: `n${src}`,
      target: `n${tgt}`,
      label: ["OWNS", "WORKS_AT", "PAID"][i % 3],
    });
  }
  return { nodes, edges };
}
