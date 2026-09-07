// The closed loop: observation -> policy -> joint position targets -> physics.
//
// This is the browser twin of microduck_rl/scripts/infer_policy.py. It runs the
// policy at 50 Hz and the physics at 200 Hz (DECIMATION substeps per control
// step), which is exactly what the checkpoints were trained against — the
// policy is blind to everything except this observation vector, so any drift
// in layout, ordering or rate shows up as a robot that flails.

import {
  ACTION_SCALE, CMD_SIZE, CTRL_DT, DECIMATION, DEFAULT_POSE, FALLEN_GZ,
  GYRO_SENSOR, JOINT_NAMES, NUM_JOINTS, OBS_SIZE, SWITCH_THRESHOLD, TRUNK_BODY,
  UPRIGHT_GZ,
} from "./microduck.ts";
import type { PolicyRunner } from "./policy.ts";
import type { Simulation } from "./scene.ts";

/**
 * What the duck is doing right now.
 *
 * - `standing`   upright, the policy holding the pose
 * - `limp`       no control at all: the fall is left to physics
 * - `settling`   targets frozen on the pose held at touchdown, mimicking the
 *                beat the real runtime waits before it tries to get up
 * - `recovering` the same policy, on its way back up
 *
 * `standing` and `recovering` run identical control — the get-up policy is
 * also a perfectly good balance controller, and on the real robot *some*
 * policy is always driving. They are separate states only so the state
 * machine (and the UI) can tell "made it" from "still trying".
 */
export type Phase = "standing" | "limp" | "settling" | "recovering";

/** 0.3 s of frozen control after a fall, as in the reference fall-detect runtime. */
const SETTLE_STEPS = 15;
/** How long the trunk has to stay tipped before a fall is called. Debounced so
 *  a shove the policy rides out does not trigger a pointless get-up. */
const FALL_DEBOUNCE_STEPS = 15;
/** A full second of upright before the get-up is declared finished. */
const UPRIGHT_STEPS = 50;
/** Give up and let the caller decide after 6 s of trying. */
const GIVEUP_STEPS = 300;

export interface Telemetry {
  phase: Phase;
  /** Projected-gravity z: -1 upright, 0 on its side, +1 upside down. */
  gravityZ: number;
  /** Trunk height above the floor, m. */
  height: number;
  simTime: number;
  /** Control steps spent in the current phase. */
  phaseSteps: number;
  /** True while the walking policy is driving. */
  walking: boolean;
}

export class MicroduckController {
  private readonly qposAdr: number[];
  private readonly dofAdr: number[];
  private readonly gyroAdr: number;
  private readonly trunkId: number;

  private readonly obs = new Float32Array(OBS_SIZE);
  private readonly cmd = new Float32Array(CMD_SIZE);
  private readonly lastAction = new Float32Array(NUM_JOINTS);

  private readonly gravity = new Float32Array(3);
  /** Control steps of shove left to apply; see push(). */
  private pushSteps = 0;

  private phase: Phase = "standing";
  private phaseSteps = 0;
  private uprightSteps = 0;
  private fallenSteps = 0;
  private simTime = 0;

  /** Fires when a get-up attempt ends: `true` if the duck is back on its feet. */
  onRecoveryEnd: ((succeeded: boolean) => void) | null = null;

  /**
   * Drive command [vx, vy, vyaw], written by the remote. Read once per control
   * step: it selects the policy and fills the command block of the observation.
   */
  readonly twist = new Float32Array(3);

  private readonly sim: Simulation;
  /** Balance and get-up policy: drives whenever the twist is zero, and always
   *  during a recovery, where it was trained on an all-zero command. */
  private policy: PolicyRunner;
  /** Locomotion policy: drives while standing with a non-zero twist. Null when
   *  the app has none, which leaves the remote inert. */
  private walker: PolicyRunner | null = null;
  private walking = false;

  constructor(sim: Simulation, policy: PolicyRunner) {
    this.sim = sim;
    this.policy = policy;
    const { mujoco, model } = sim;
    this.qposAdr = JOINT_NAMES.map((n) => model.jnt(n).qposadr);
    this.dofAdr = JOINT_NAMES.map((n) => model.jnt(n).dofadr);
    this.gyroAdr = model.sensor(GYRO_SENSOR).adr;
    this.trunkId = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY.value, TRUNK_BODY);
  }

  /** Projected gravity in the trunk frame: world -Z rotated into the body. */
  private projectedGravity(out: Float32Array, offset: number): void {
    const q = this.sim.data.body(this.trunkId).xquat; // [w, x, y, z]
    const [w, x, y, z] = [q[0], q[1], q[2], q[3]];
    // Conjugate rotation of (0, 0, -1), written out: cheaper and dependency
    // free compared with building a quaternion object every control step.
    out[offset + 0] = -2 * (x * z - w * y);
    out[offset + 1] = -2 * (y * z + w * x);
    out[offset + 2] = -(1 - 2 * (x * x + y * y));
  }

  get gravityZ(): number {
    this.projectedGravity(this.gravity, 0);
    return this.gravity[2];
  }

  private buildObs(): Float32Array {
    const { data } = this.sim;
    const { qpos, qvel, sensordata } = data;
    const obs = this.obs;
    let i = 0;
    for (let a = 0; a < 3; a++) obs[i++] = sensordata[this.gyroAdr + a];
    this.projectedGravity(obs, i); i += 3;
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qpos[this.qposAdr[j]] - DEFAULT_POSE[j];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = qvel[this.dofAdr[j]];
    for (let j = 0; j < NUM_JOINTS; j++) obs[i++] = this.lastAction[j];
    // cmd = [twist(3) | head(4) | body(6)]. The standing policy was trained on
    // all zeros; the walking one reads the twist and zeros in the other slots.
    this.cmd.fill(0);
    if (this.walking) this.cmd.set(this.twist);
    for (let c = 0; c < CMD_SIZE; c++) obs[i++] = this.cmd[c];
    return obs;
  }

  /**
   * The runner for this step. Walking wants the duck upright and a twist past
   * the switch threshold; everything else is the standing policy. Swapping
   * runners clears lastAction, which belongs to the other policy's action
   * distribution and is part of what the new one observes.
   */
  private select(): PolicyRunner {
    const [vx, vy, vyaw] = this.twist;
    const magnitude = Math.hypot(vx, vy, vyaw);
    const walker = this.phase === "standing" && magnitude > SWITCH_THRESHOLD ? this.walker : null;
    const walking = walker !== null;
    if (walking !== this.walking) {
      this.walking = walking;
      this.lastAction.fill(0);
    }
    return walker ?? this.policy;
  }

  private stepPhysics(): void {
    const { mujoco, model, data } = this.sim;
    for (let s = 0; s < DECIMATION; s++) mujoco.mj_step(model, data);
    this.simTime += CTRL_DT;
  }

  /** One 50 Hz control step. Async because ONNX inference is. */
  async step(): Promise<void> {
    const { data } = this.sim;
    this.phaseSteps++;

    if (this.phase === "recovering" || this.phase === "standing") {
      const action = await this.select().run(this.buildObs());
      this.lastAction.set(action);
      for (let j = 0; j < NUM_JOINTS; j++) {
        data.ctrl[j] = DEFAULT_POSE[j] + action[j] * ACTION_SCALE;
      }
    } else if (this.phase === "limp") {
      // Zero torque: the position actuators still pull toward ctrl, so the
      // limp fall is approximated by tracking the pose the duck is already in.
      for (let j = 0; j < NUM_JOINTS; j++) data.ctrl[j] = data.qpos[this.qposAdr[j]];
    }
    // "settling" holds whatever ctrl was last written.

    if (this.pushSteps > 0 && --this.pushSteps === 0) {
      this.sim.data.xfrc_applied.fill(0, this.trunkId * 6, this.trunkId * 6 + 6);
    }
    this.stepPhysics();
    this.advance();
  }

  private advance(): void {
    const gz = this.gravityZ;
    switch (this.phase) {
      case "limp":
        // Wait for the tumble to actually put the duck down before starting
        // the settle beat, so a shove that it rides out does not trigger one.
        if (gz > FALLEN_GZ) this.setPhase("settling");
        break;
      case "settling":
        if (this.phaseSteps >= SETTLE_STEPS) {
          this.lastAction.fill(0);
          this.setPhase("recovering");
        }
        break;
      case "recovering":
        this.uprightSteps = gz < UPRIGHT_GZ ? this.uprightSteps + 1 : 0;
        if (this.uprightSteps >= UPRIGHT_STEPS) {
          this.setPhase("standing");
          this.onRecoveryEnd?.(true);
        } else if (this.phaseSteps >= GIVEUP_STEPS) {
          this.setPhase("standing");
          this.onRecoveryEnd?.(false);
        }
        break;
      case "standing":
        // Fall detection, as on the real robot: the policy is left in charge
        // until the trunk has genuinely been down for a moment.
        this.fallenSteps = gz > FALLEN_GZ ? this.fallenSteps + 1 : 0;
        if (this.fallenSteps >= FALL_DEBOUNCE_STEPS) this.setPhase("settling");
        break;
    }
  }

  private setPhase(phase: Phase): void {
    this.phase = phase;
    this.phaseSteps = 0;
    this.fallenSteps = 0;
    if (phase !== "recovering") this.uprightSteps = 0;
  }

  /** Swap the driving policy without disturbing the simulation. lastAction is
   *  cleared because it is part of the observation and belonged to the old
   *  policy's action distribution. */
  setPolicy(policy: PolicyRunner): void {
    this.policy = policy;
    this.lastAction.fill(0);
  }

  /** Install the locomotion policy the remote drives. */
  setWalker(walker: PolicyRunner | null): void {
    this.walker = walker;
  }

  get policyLabel(): string {
    return this.policy.label;
  }

  /** Start a get-up attempt from wherever the duck currently is. */
  standUp(): void {
    if (this.phase === "recovering") return;
    this.lastAction.fill(0);
    this.setPhase("recovering");
  }

  /** Cut control and let the duck fall; the settle + get-up follow on their own. */
  goLimp(): void {
    this.setPhase("limp");
  }

  /**
   * Shove the trunk from a random direction. The force is held for a few
   * control steps (counted in step(), so the impulse does not depend on the
   * wall-clock frame rate) and then released.
   *
   * The policy keeps driving throughout: a light shove gets absorbed, a hard
   * one topples the duck and the fall detection above takes it from there.
   */
  push(strength = 1): void {
    const f = this.sim.data.xfrc_applied;
    const base = this.trunkId * 6;
    const angle = Math.random() * Math.PI * 2;
    f[base + 0] = Math.cos(angle) * 3.5 * strength;
    f[base + 1] = Math.sin(angle) * 3.5 * strength;
    f[base + 2] = 1.5 * strength;
    this.pushSteps = 3; // 60 ms of contact
  }

  /** Back to the STAND keyframe, upright and holding the reference pose. */
  reset(): void {
    const { mujoco, model, data, standKey } = this.sim;
    mujoco.mj_resetDataKeyframe(model, data, standKey);
    data.xfrc_applied.fill(0);
    mujoco.mj_forward(model, data);
    this.pushSteps = 0;
    this.lastAction.fill(0);
    this.simTime = 0;
    this.setPhase("standing");
  }

  /**
   * Drop the duck onto the floor in a random tumbled pose and go limp, so the
   * next get-up starts from a genuine fallen state rather than a hand-authored
   * keyframe.
   */
  knockDown(): void {
    const { mujoco, model, data, standKey } = this.sim;
    mujoco.mj_resetDataKeyframe(model, data, standKey);
    data.xfrc_applied.fill(0);
    this.pushSteps = 0;
    // Tip well past horizontal about a random axis in the ground plane and
    // drop from 12 cm: it always lands, but never twice the same way.
    const axis = Math.random() * Math.PI * 2;
    const tilt = (Math.PI / 2) * (1 + Math.random() * 0.7); // 90-153 deg
    const s = Math.sin(tilt / 2);
    data.qpos[2] = 0.12;
    data.qpos[3] = Math.cos(tilt / 2);
    data.qpos[4] = Math.cos(axis) * s;
    data.qpos[5] = Math.sin(axis) * s;
    data.qpos[6] = 0;
    mujoco.mj_forward(model, data);
    this.lastAction.fill(0);
    this.setPhase("limp");
  }

  telemetry(): Telemetry {
    return {
      phase: this.phase,
      gravityZ: this.gravityZ,
      height: this.sim.data.qpos[2],
      simTime: this.simTime,
      phaseSteps: this.phaseSteps,
      walking: this.walking,
    };
  }
}
