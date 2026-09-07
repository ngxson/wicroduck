<script lang="ts">
  // The gamepad, on the page. Left stick y is forward speed and left stick x
  // is yaw rate, like a car; the right stick x is strafe. Each axis is scaled
  // by the full-deflection maximum microduck/padd uses. The keyboard doubles
  // the sticks at full deflection while a key is held.

  import { MAX_ANGULAR, MAX_LINEAR } from "../sim/microduck.ts";
  import { type Session } from "./session.svelte";
  import Stick from "./Stick.svelte";

  const { session }: { session: Session } = $props();

  let leftX = $state(0);
  let leftY = $state(0);
  let rightX = $state(0);

  /** Keys currently held, by physical code, so the layout does not matter. */
  const held = new Set<string>();
  let keyX = $state(0);
  let keyY = $state(0);
  let keyYaw = $state(0);

  const KEY_FORWARD = "ArrowUp";
  const KEY_BACKWARD = "ArrowDown";
  const KEY_YAW_LEFT = "ArrowLeft";
  const KEY_YAW_RIGHT = "ArrowRight";
  const KEY_LEFT = "KeyQ";
  const KEY_RIGHT = "KeyE";
  const KEYS = new Set([KEY_FORWARD, KEY_BACKWARD, KEY_YAW_LEFT, KEY_YAW_RIGHT, KEY_LEFT, KEY_RIGHT]);

  /** A held key is a stick at full deflection; a stick being dragged wins otherwise. */
  const axisY = $derived(keyY !== 0 ? keyY : leftY);
  const axisYaw = $derived(keyYaw !== 0 ? keyYaw : leftX);
  const axisX = $derived(keyX !== 0 ? keyX : rightX);

  // Stick left reads negative on a pad and the robot's y is positive to the
  // left, hence the sign flips on strafe and yaw.
  $effect(() => {
    session.setTwist(axisY * MAX_LINEAR, -axisX * MAX_LINEAR, -axisYaw * MAX_ANGULAR);
  });

  function readKeys(): void {
    const axis = (neg: string, pos: string): number => (held.has(pos) ? 1 : 0) - (held.has(neg) ? 1 : 0);
    keyY = axis(KEY_BACKWARD, KEY_FORWARD);
    keyYaw = axis(KEY_YAW_LEFT, KEY_YAW_RIGHT);
    keyX = axis(KEY_LEFT, KEY_RIGHT);
  }

  function typing(e: KeyboardEvent): boolean {
    const t = e.target;
    return t instanceof HTMLElement && (t.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(t.tagName));
  }

  function keydown(e: KeyboardEvent): void {
    if (!session.ready || !KEYS.has(e.code) || typing(e)) return;
    e.preventDefault();
    held.add(e.code);
    readKeys();
  }

  function keyup(e: KeyboardEvent): void {
    if (!held.delete(e.code)) return;
    readKeys();
  }

  /** Focus lost mid-press never reaches keyup, so drop everything held. */
  function blur(): void {
    held.clear();
    readKeys();
  }
</script>

<svelte:window onkeydown={keydown} onkeyup={keyup} onblur={blur} />

<div class="remote">
  <Stick
    label="Drive"
    disabled={!session.ready}
    x={keyYaw}
    y={keyY}
    onmove={(x, y) => { leftX = x; leftY = y; }}
  />
  <Stick
    label="Strafe"
    horizontal
    disabled={!session.ready}
    x={keyX}
    onmove={(x) => { rightX = x; }}
  />
</div>
<p class="status" class:walking={session.walking}>
  {session.walking ? "Walking" : "Standing"}
  <small>Arrows drive, Q/E strafe</small>
</p>

<style>
  .remote { display: flex; justify-content: space-around; align-items: center; gap: 10px; }
  .status {
    margin: 0; display: flex; justify-content: space-between; align-items: baseline;
    font-size: 11px; font-weight: 600; color: var(--muted);
  }
  .status.walking { color: var(--accent); }
  .status small { font-size: 10px; font-weight: 400; color: var(--muted); }
</style>
