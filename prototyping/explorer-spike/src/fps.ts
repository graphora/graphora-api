// Rolling FPS meter — reports average frame time over the last N frames
// using requestAnimationFrame. Subscribes via callback so React can
// render the number without tying itself to rAF.

export function createFpsMeter(windowSize = 60, onUpdate: (fps: number) => void) {
  const timestamps: number[] = [];
  let raf = 0;
  let last = performance.now();

  function frame(now: number) {
    const delta = now - last;
    last = now;
    timestamps.push(delta);
    if (timestamps.length > windowSize) timestamps.shift();
    if (timestamps.length === windowSize) {
      const avg = timestamps.reduce((a, b) => a + b, 0) / timestamps.length;
      onUpdate(Math.round(1000 / avg));
    }
    raf = requestAnimationFrame(frame);
  }

  return {
    start() {
      last = performance.now();
      raf = requestAnimationFrame(frame);
    },
    stop() {
      cancelAnimationFrame(raf);
      timestamps.length = 0;
    },
  };
}
