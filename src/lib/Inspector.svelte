<script lang="ts">
  import { CTRL_DT, OBS_SIZE, NUM_JOINTS, TIMESTEP } from "../sim/microduck.ts";
  import { type Session } from "./session.svelte";
  import Remote from "./Remote.svelte";

  const { session }: { session: Session } = $props();

  const controlHz = Math.round(1 / CTRL_DT);
  const physicsHz = Math.round(1 / TIMESTEP);
</script>

<aside class="inspector">
  <section>
    <h2>Policy</h2>
    <label class="picker">
      <select
        value={session.activePolicy}
        disabled={!session.ready || session.policyBusy}
        onchange={(e) => void session.selectPolicy(e.currentTarget.value)}
      >
        {#each session.policies as p (p.id)}
          <option value={p.id}>{p.label}{p.kind === "checkpoint" ? " (trained here)" : ""}</option>
        {/each}
      </select>
    </label>
    {#if session.policyError}
      <p class="err">{session.policyError}</p>
    {:else if session.policies.length === 1}
      <p class="hint">Train a policy and save it to run it here.</p>
    {/if}
    <dl>
      <div><dt>Interface</dt><dd class="mono">{OBS_SIZE} → {NUM_JOINTS}</dd></div>
      <div><dt>Control</dt><dd class="mono">{controlHz} Hz</dd></div>
      <div><dt>Physics</dt><dd class="mono">{physicsHz} Hz</dd></div>
    </dl>
  </section>

  <section>
    <h2>Remote</h2>
    <Remote {session} />
  </section>

  <section>
    <h2>Actions</h2>
    <div class="stack">
      <button class="primary" disabled={!session.ready} onclick={() => session.knockDown()}>
        Knock down &amp; stand up
      </button>
      <div class="row">
        <button disabled={!session.ready} onclick={() => session.push()}>Push</button>
        <button disabled={!session.ready} onclick={() => session.standUp()}>Stand up</button>
      </div>
      <button disabled={!session.ready} onclick={() => session.reset()}>Reset</button>
    </div>
  </section>

  <section>
    <h2>Options</h2>
    <label class="toggle">
      <!-- Not `bind:` + a separate onchange: the two handlers race, and the
           side effect would read a stale value if it won. One handler owns it. -->
      <input
        type="checkbox"
        checked={session.autoRepeat}
        onchange={(e) => session.setAutoRepeat(e.currentTarget.checked)}
        disabled={!session.ready}
      />
      <span>Auto-repeat<small>Knock over again after each get-up</small></span>
    </label>
    <label class="toggle">
      <input type="checkbox" bind:checked={session.showCollision} disabled={!session.ready} />
      <span>Collision geoms<small>Overlay the physics proxies</small></span>
    </label>
  </section>

  <p class="note">Drag to orbit, scroll to zoom. The camera follows the trunk.</p>
</aside>

<style>
  .inspector {
    grid-area: aside;
    display: flex; flex-direction: column; gap: 18px;
    padding: 14px; overflow-y: auto;
    background: var(--panel); border-left: 1px solid var(--line);
  }

  section { display: flex; flex-direction: column; gap: 9px; }
  h2 {
    margin: 0; font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em; color: var(--muted);
  }

  dl { margin: 0; display: flex; flex-direction: column; gap: 5px; font-size: 12px; }
  dl div { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; }
  dt { color: var(--muted); }
  dd { margin: 0; text-align: right; overflow-wrap: anywhere; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 11px; }

  .picker select {
    width: 100%; font: inherit; font-size: 11px; color: var(--ink);
    background: var(--panel-hi); border: 1px solid var(--line);
    border-radius: 6px; padding: 5px 6px;
  }
  .picker select:disabled { opacity: 0.5; }
  .hint, .err { margin: 0; font-size: 10px; line-height: 1.5; color: var(--muted); }
  .err { color: var(--danger); }

  .stack { display: flex; flex-direction: column; gap: 6px; }
  .row { display: flex; gap: 6px; }
  .row button { flex: 1; }

  button {
    font: inherit; font-size: 12px; font-weight: 600;
    color: var(--ink); background: var(--panel-hi);
    border: 1px solid var(--line); border-radius: 7px;
    padding: 8px 10px; cursor: pointer;
  }
  button:hover:not(:disabled) { border-color: var(--line-hi); background: #1c2230; }
  button:disabled { opacity: 0.4; cursor: not-allowed; }
  button.primary { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
  button.primary:hover:not(:disabled) { background: var(--accent-hi); border-color: var(--accent-hi); }

  .toggle { display: flex; align-items: flex-start; gap: 8px; font-size: 12px; cursor: pointer; }
  .toggle input { margin: 1px 0 0; accent-color: var(--accent); cursor: pointer; }
  .toggle span { display: flex; flex-direction: column; line-height: 1.3; }
  .toggle small { font-size: 10px; color: var(--muted); }
  .toggle:has(input:disabled) { opacity: 0.4; cursor: not-allowed; }

  .note {
    margin: auto 0 0; padding-top: 12px;
    border-top: 1px solid var(--line);
    font-size: 10px; line-height: 1.5; color: var(--muted);
  }

  @media (max-width: 1040px) {
    .inspector {
      border-left: 0; border-top: 1px solid var(--line);
      flex-direction: row; flex-wrap: wrap; gap: 20px;
    }
    section { flex: 1 1 200px; }
    .note { display: none; }
  }
</style>
