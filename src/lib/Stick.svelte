<script lang="ts">
  // One analogue stick: a pad the pointer drags a knob around. Reports its
  // deflection as gamepad axes, x right and y up in [-1, 1], with the per-axis
  // deadzone a real pad daemon applies. Releasing the pointer springs the knob
  // back to centre and reports zero, so letting go always means stop.

  import { STICK_DEADZONE } from "../sim/microduck.ts";

  interface Props {
    label: string;
    /** Lock the knob to the horizontal axis; y is then always zero. */
    horizontal?: boolean;
    disabled?: boolean;
    /** Deflection the stick is being driven to from elsewhere, drawn but not
     *  reported back. Lets a keyboard press show on the pad. */
    x?: number;
    y?: number;
    onmove: (x: number, y: number) => void;
  }

  const { label, horizontal = false, disabled = false, x = 0, y = 0, onmove }: Props = $props();

  /** Pad radius, matching the .pad width below. */
  const PAD_RADIUS = 48;
  /** Knob travel as a fraction of the pad radius. */
  const TRAVEL = 0.62;
  const KNOB_PX = PAD_RADIUS * TRAVEL;

  let pad: HTMLDivElement;
  let dragX = $state(0);
  let dragY = $state(0);
  let dragging = $state(false);

  const showX = $derived(dragging ? dragX : x);
  const showY = $derived(dragging ? dragY : y);

  function deadzone(v: number): number {
    return Math.abs(v) < STICK_DEADZONE ? 0 : v;
  }

  function update(e: PointerEvent): void {
    const r = pad.getBoundingClientRect();
    const radius = r.width / 2;
    let dx = (e.clientX - (r.left + radius)) / (radius * TRAVEL);
    let dy = -(e.clientY - (r.top + radius)) / (radius * TRAVEL);
    if (horizontal) dy = 0;
    const m = Math.hypot(dx, dy);
    if (m > 1) {
      dx /= m;
      dy /= m;
    }
    dragX = dx;
    dragY = dy;
    onmove(deadzone(dx), deadzone(dy));
  }

  function down(e: PointerEvent): void {
    if (disabled) return;
    pad.setPointerCapture(e.pointerId);
    dragging = true;
    update(e);
  }

  function move(e: PointerEvent): void {
    if (dragging) update(e);
  }

  function up(): void {
    if (!dragging) return;
    dragging = false;
    dragX = 0;
    dragY = 0;
    onmove(0, 0);
  }
</script>

<div class="stick" class:disabled>
  <div
    class="pad"
    class:horizontal
    role="slider"
    aria-label={label}
    aria-valuenow={Math.round(showX * 100)}
    tabindex="-1"
    bind:this={pad}
    onpointerdown={down}
    onpointermove={move}
    onpointerup={up}
    onpointercancel={up}
  >
    <div class="cross"></div>
    <div
      class="knob"
      class:live={dragging || showX !== 0 || showY !== 0}
      style="transform: translate({showX * KNOB_PX}px, {-showY * KNOB_PX}px)"
    ></div>
  </div>
  <span class="label">{label}</span>
</div>

<style>
  .stick { display: flex; flex-direction: column; align-items: center; gap: 5px; }
  .disabled { opacity: 0.4; }

  .pad {
    position: relative; width: 96px; height: 96px;
    border-radius: 50%; background: var(--panel-hi);
    border: 1px solid var(--line); touch-action: none; cursor: grab;
    user-select: none;
  }
  .pad.horizontal { height: 40px; border-radius: 20px; }
  .disabled .pad { cursor: not-allowed; }

  .cross {
    position: absolute; inset: 0; pointer-events: none;
    background:
      linear-gradient(var(--line), var(--line)) center / 1px 60% no-repeat,
      linear-gradient(var(--line), var(--line)) center / 60% 1px no-repeat;
  }
  .horizontal .cross { background: linear-gradient(var(--line), var(--line)) center / 60% 1px no-repeat; }

  .knob {
    position: absolute; left: 50%; top: 50%; width: 34px; height: 34px;
    margin: -17px 0 0 -17px; border-radius: 50%;
    background: var(--muted); border: 1px solid var(--line-hi);
    pointer-events: none;
  }
  .horizontal .knob { width: 28px; height: 28px; margin: -14px 0 0 -14px; }
  .knob.live { background: var(--accent); border-color: var(--accent-hi); }

  .label { font-size: 10px; color: var(--muted); }
</style>
