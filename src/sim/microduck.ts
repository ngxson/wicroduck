// Everything that is specific to the Microduck robot and to the policy
// interface the shipped ONNX checkpoints were trained against.
//
// The numbers here are not free parameters: they mirror
// microduck_rl/scripts/infer_policy.py (the reference deployment loop) and the
// metadata baked into the ONNX files. Changing one without retraining silently
// produces a robot that twitches instead of standing.

/** Actuator order in the MJCF, and the order the policy expects. */
export const JOINT_NAMES = [
  "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
  "neck_pitch", "head_pitch", "head_yaw", "head_roll",
  "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
] as const;

export const NUM_JOINTS = JOINT_NAMES.length; // 14

/**
 * STAND2 reference pose (`default_joint_pos` in the ONNX metadata). Actions are
 * offsets from it and joint observations are relative to it, so it is both the
 * zero of the action space and the zero of the observation space.
 */
export const DEFAULT_POSE = new Float32Array([
  0, -0.08726646259971647, -0.457924, -0.00494, 0.452984,
  0.3490658503988659, 0.3490658503988659, 0, 0,
  0, 0.08726646259971647, 0.457924, 0.00494, -0.452984,
]);

/** obs = [base_ang_vel(3), projected_gravity(3), joint_pos(14), joint_vel(14),
 *         actions(14), command(13)] — 61 floats. */
export const CMD_SIZE = 13;
export const OBS_SIZE = 3 + 3 + NUM_JOINTS * 3 + CMD_SIZE; // 61
/** Where the command block starts inside one observation. The get-up policies
 *  were trained with it all zeros; tasks that need to tell the policy
 *  something — a reference-motion phase, a velocity target — write here. */
export const CMD_OFFSET = OBS_SIZE - CMD_SIZE; // 48

/**
 * Hinge limits from the MJCF, policy order, radians: [min, max] per joint.
 *
 * Duplicated here rather than read off the compiled model because the motion
 * parser has to reject an out-of-range pose before any model exists — and a
 * pose MuJoCo silently clamps is a motion that plays back wrong for reasons
 * nothing reports. scripts/check-motion.ts asserts these still match the
 * compiled ranges, so the copy cannot drift.
 */
export const JOINT_LIMITS: readonly (readonly [number, number])[] = [
  [-0.4363, 0.5236],   // left_hip_yaw
  [-0.3840, 0.3840],   // left_hip_roll
  [-1.5708, 1.5708],   // left_hip_pitch
  [-1.5708, 1.5708],   // left_knee
  [-1.5708, 1.5708],   // left_ankle
  [-1.5708, 1.0472],   // neck_pitch
  [-1.5708, 1.5708],   // head_pitch
  [-2.9671, 2.9671],   // head_yaw
  [-0.4363, 0.4363],   // head_roll
  [-0.5236, 0.4363],   // right_hip_yaw
  [-0.3840, 0.3840],   // right_hip_roll
  [-1.5708, 1.5708],   // right_hip_pitch
  [-1.5708, 1.5708],   // right_knee
  [-1.5708, 1.5708],   // right_ankle
];

export const ACTION_SCALE = 1.0;

/**
 * Drive command, as the real robot's gamepad daemon (microduck/padd) shapes it:
 * cmd[0..3] = [vx forward m/s, vy left m/s, vyaw rad/s], each a stick axis
 * scaled by its full-deflection maximum. Deflection below the deadzone reads
 * as centre, so a resting stick never makes the duck creep.
 */
export const MAX_LINEAR = 1.0;
export const MAX_ANGULAR = 3.0;
export const STICK_DEADZONE = 0.1;
/** Twist magnitude above which the walking policy drives instead of the
 *  standing one, from infer_policy.py's switch_threshold. */
export const SWITCH_THRESHOLD = 0.05;

/** Physics runs at 200 Hz, the policy at 50 Hz — the training decimation. */
export const TIMESTEP = 0.005;
export const DECIMATION = 4;
export const CTRL_DT = TIMESTEP * DECIMATION;

/** Trunk body carrying the free joint and the IMU site. */
export const TRUNK_BODY = "trunk_base";
export const GYRO_SENSOR = "imu_ang_vel";

/**
 * Upright test on the projected-gravity z component (obs[5]): -1 is perfectly
 * upright, 0 is on its side. Thresholds are the ones the reference fall-detect
 * runtime uses: past -0.5 counts as fallen, below -0.85 counts as recovered.
 */
export const FALLEN_GZ = -0.5;
export const UPRIGHT_GZ = -0.85;
