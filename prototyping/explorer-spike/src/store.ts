import { create } from "zustand";
import { generateFixture, SpikeGraph } from "./fixture";

// Single tiny store — enough to prove zustand is in the bundle and
// that fixture regeneration doesn't cause React gymnastics. Real
// Explorer state lives elsewhere; this just drives the spike UI.
interface SpikeStore {
  nodeCount: number;
  fps: number;
  graph: SpikeGraph;
  setNodeCount: (n: number) => void;
  setFps: (f: number) => void;
}

export const useSpikeStore = create<SpikeStore>((set) => ({
  nodeCount: 1000,
  fps: 0,
  graph: generateFixture(1000),
  setNodeCount: (nodeCount) =>
    set({ nodeCount, graph: generateFixture(nodeCount) }),
  setFps: (fps) => set({ fps }),
}));
