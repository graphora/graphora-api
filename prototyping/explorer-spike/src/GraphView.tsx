import { useEffect, useRef } from "react";
import cytoscape, { Core, ElementDefinition } from "cytoscape";
import { SpikeGraph } from "./fixture";

interface Props {
  graph: SpikeGraph;
}

// Mount Cytoscape once per graph. `data` updates that don't swap the
// graph shape go through cy.data / cy.json; full-graph swaps rebuild
// the elements. Layout is cose (force-directed) — switch to preset
// if initial-layout cost dominates measured render time.
export function GraphView({ graph }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const cyRef = useRef<Core | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const elements: ElementDefinition[] = [
      ...graph.nodes.map((n) => ({
        data: { id: n.id, label: n.label, type: n.type },
      })),
      ...graph.edges.map((e) => ({
        data: { id: e.id, source: e.source, target: e.target, label: e.label },
      })),
    ];

    const cy = cytoscape({
      container: containerRef.current,
      elements,
      // Large fixtures collapse cose almost instantly into a hairball;
      // use a grid initial layout to keep startup cost bounded.
      // Real Explorer will use cose-bilkent or a worker-based layout;
      // the spike only cares about render+interaction, not beauty.
      layout: {
        name: graph.nodes.length > 2000 ? "grid" : "cose",
        fit: true,
      },
      style: [
        {
          selector: "node",
          style: {
            "background-color": "#3b82f6",
            "width": 8,
            "height": 8,
            "label": "data(label)",
            "font-size": 6,
            "color": "#64748b",
            "text-opacity": 0.6,
          },
        },
        {
          selector: "edge",
          style: {
            "width": 0.5,
            "line-color": "#cbd5e1",
            "curve-style": "haystack", // cheapest edge renderer
            "opacity": 0.5,
          },
        },
      ],
      // Perf knobs — these mirror what the real Explorer will set.
      hideEdgesOnViewport: true,
      textureOnViewport: true,
      motionBlur: false,
      pixelRatio: 1,
      wheelSensitivity: 0.2,
    });

    cyRef.current = cy;

    return () => {
      cy.destroy();
      cyRef.current = null;
    };
  }, [graph]);

  return <div ref={containerRef} style={{ width: "100%", height: "100%" }} />;
}
