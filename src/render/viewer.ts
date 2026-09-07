// A generic MuJoCo -> three.js view: it builds one three.js mesh per MuJoCo
// geom straight from the compiled model's arrays, then each frame copies
// data.geom_xpos / data.geom_xmat onto them. Nothing here knows about the
// duck, so it renders any MJCF the sim module compiles.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GEOM, type MjData, type MjModel } from "../sim/mujoco.ts";

/** MuJoCo is z-up, three.js is y-up. One rotation on the root holds the
 *  whole scene instead of converting every pose. */
const Z_UP = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), -Math.PI / 2);

/** MuJoCo geom groups: 2 is the visual shell, 3 the collision proxies. The
 *  all-collisions model carries both, at the same poses. */
export const GROUP_VISUAL = 2;
export const GROUP_COLLISION = 3;

function meshGeometry(model: MjModel, meshId: number): THREE.BufferGeometry {
  const vertAdr = model.mesh_vertadr[meshId];
  const vertNum = model.mesh_vertnum[meshId];
  const faceAdr = model.mesh_faceadr[meshId];
  const faceNum = model.mesh_facenum[meshId];

  const position = new Float32Array(vertNum * 3);
  const normal = new Float32Array(vertNum * 3);
  for (let i = 0; i < vertNum * 3; i++) {
    position[i] = model.mesh_vert[vertAdr * 3 + i];
    normal[i] = model.mesh_normal[vertAdr * 3 + i];
  }
  const index = new Uint32Array(faceNum * 3);
  for (let i = 0; i < faceNum * 3; i++) index[i] = model.mesh_face[faceAdr * 3 + i];

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(position, 3));
  // MuJoCo emits one normal per vertex for these meshes (mesh_normalnum ==
  // mesh_vertnum and mesh_facenormal == mesh_face), so the normals can be
  // indexed exactly like the positions instead of unwelding every face.
  geo.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
  geo.setIndex(new THREE.BufferAttribute(index, 1));
  return geo;
}

/** Time constant of the camera follow, s. Frame-rate independent, so the
 *  gait's trunk sway is smoothed the same at 60 Hz and 240 Hz. */
const FOLLOW_TAU = 0.12;

/** Checker cell edge on the floor, m. */
const CHECKER_CELL = 0.1;
/** How much lighter the alternate cell is than the floor's own color. */
const CHECKER_LIFT = 0.05;

/**
 * A 2x2 checker in the floor's color and a lighter shade of it, repeated over
 * the plane. Mipmapped and anisotropic, so at distance it settles into the
 * flat mean gray instead of shimmering.
 */
function checkerTexture(r: number, g: number, b: number, halfSize: number, anisotropy: number): THREE.Texture {
  const lift = (v: number) => Math.min(1, v + CHECKER_LIFT);
  const dark = [r, g, b].map((v) => Math.round(v * 255));
  const light = [r, g, b].map((v) => Math.round(lift(v) * 255));
  const data = new Uint8Array([...dark, 255, ...light, 255, ...light, 255, ...dark, 255]);
  const tex = new THREE.DataTexture(data, 2, 2);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(halfSize / CHECKER_CELL, halfSize / CHECKER_CELL);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.generateMipmaps = true;
  tex.anisotropy = anisotropy;
  tex.needsUpdate = true;
  return tex;
}

/** Primitive geoms, sized so a unit build can be scaled by geom_size. */
function primitiveGeometry(type: number, size: Float64Array | number[]): THREE.BufferGeometry | null {
  switch (type) {
    case GEOM.PLANE:
      // MuJoCo planes face local +Z, which is what an unrotated three.js
      // PlaneGeometry already does — the z-up flip lives on the scene root.
      // A size of 0 means "infinite"; a big quad reads as a floor either way.
      return new THREE.PlaneGeometry((size[0] || 10) * 2, (size[1] || 10) * 2, 1, 1);
    case GEOM.SPHERE:
      return new THREE.SphereGeometry(size[0], 24, 16);
    case GEOM.CAPSULE:
      return new THREE.CapsuleGeometry(size[0], size[1] * 2, 6, 16).rotateX(Math.PI / 2);
    case GEOM.ELLIPSOID:
      return new THREE.SphereGeometry(1, 24, 16).scale(size[0], size[1], size[2]);
    case GEOM.CYLINDER:
      return new THREE.CylinderGeometry(size[0], size[0], size[1] * 2, 24).rotateX(Math.PI / 2);
    case GEOM.BOX:
      return new THREE.BoxGeometry(size[0] * 2, size[1] * 2, size[2] * 2);
    default:
      return null; // heightfields and anything else: not rendered
  }
}

/** One physics step's worth of geom poses and the followed body's position. */
interface Pose {
  xpos: Float64Array;
  xmat: Float64Array;
  follow: Float64Array;
}

export interface ViewerOptions {
  /** Follow the trunk with the camera target instead of orbiting a fixed point. */
  followBody?: number;
}

export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  readonly controls: OrbitControls;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly root = new THREE.Object3D();
  /** three.js object per MuJoCo geom id; null where the geom is not drawn. */
  private readonly geomObjects: (THREE.Mesh | null)[] = [];
  private readonly collisionObjects: THREE.Mesh[] = [];
  /** One geometry per mesh asset, shared by every geom referencing it; owned
   *  here so it is disposed once rather than once per geom. */
  private meshCache = new Map<number, THREE.BufferGeometry>();
  private prev: Pose = { xpos: new Float64Array(0), xmat: new Float64Array(0), follow: new Float64Array(3) };
  private curr: Pose = { xpos: new Float64Array(0), xmat: new Float64Array(0), follow: new Float64Array(3) };
  private readonly followTarget = new THREE.Vector3();
  private readonly followDelta = new THREE.Vector3();
  private followBody: number | null = null;
  /** Shadow-casting light, kept over the followed body so the tight shadow
   *  frustum travels with the robot. */
  private readonly key: THREE.DirectionalLight;
  private readonly keyOffset = new THREE.Vector3(0.6, 1.2, 0.5);
  private readonly resizeObserver: ResizeObserver;
  private readonly mat = new THREE.Matrix4();

  private readonly canvas: HTMLCanvasElement;

  constructor(canvas: HTMLCanvasElement) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    this.scene.background = new THREE.Color(0x0f1116);
    this.scene.fog = new THREE.Fog(0x0f1116, 1.5, 6);

    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
    this.camera.position.set(0.55, 0.28, 0.55);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 0.08, 0);
    this.controls.minDistance = 0.15;
    this.controls.maxDistance = 4;

    this.root.quaternion.copy(Z_UP);
    this.scene.add(this.root);

    const hemi = new THREE.HemisphereLight(0x9fb4d0, 0x20242c, 1.5);
    this.scene.add(hemi);
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.copy(this.keyOffset);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    // Tight ortho frustum: the whole robot is ~25 cm tall, so the default
    // camera box would spend its entire depth range on empty space.
    const s = 0.6;
    key.shadow.camera.left = -s; key.shadow.camera.right = s;
    key.shadow.camera.top = s; key.shadow.camera.bottom = -s;
    key.shadow.camera.near = 0.1; key.shadow.camera.far = 4;
    key.shadow.bias = -0.0015;
    this.scene.add(key);
    this.scene.add(key.target);
    this.key = key;
    this.scene.add(new THREE.DirectionalLight(0xbcd4ff, 0.5).translateX(-1).translateY(0.5));

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(canvas.parentElement ?? canvas);
    this.resize();
  }

  /** (Re)build the render objects for a compiled model. */
  build(model: MjModel, options: ViewerOptions = {}): void {
    this.disposeObjects();
    this.root.clear();
    this.followBody = options.followBody ?? null;
    const meshCache = this.meshCache;

    for (let g = 0; g < model.ngeom; g++) {
      const type = model.geom_type[g];
      const group = model.geom_group[g];
      const size = [model.geom_size[g * 3], model.geom_size[g * 3 + 1], model.geom_size[g * 3 + 2]];

      let geometry: THREE.BufferGeometry | null;
      let meshId = -1;
      if (type === GEOM.MESH) {
        meshId = model.geom_dataid[g];
        geometry = meshCache.get(meshId) ?? null;
        if (!geometry) {
          geometry = meshGeometry(model, meshId);
          meshCache.set(meshId, geometry);
        }
      } else {
        geometry = primitiveGeometry(type, size);
      }
      if (!geometry) {
        this.geomObjects.push(null);
        continue;
      }

      const matid = model.geom_matid[g];
      const rgba = matid >= 0 ? model.mat_rgba : model.geom_rgba;
      const base = matid >= 0 ? matid * 4 : g * 4;
      const isCollision = group === GROUP_COLLISION;
      const isPlane = type === GEOM.PLANE;
      // The floor carries its color in the checker; every other geom in the material.
      const map = isPlane
        ? checkerTexture(rgba[base], rgba[base + 1], rgba[base + 2], size[0] || 10, this.renderer.capabilities.getMaxAnisotropy())
        : null;
      const material = new THREE.MeshStandardMaterial({
        color: isPlane ? 0xffffff : new THREE.Color(rgba[base], rgba[base + 1], rgba[base + 2]),
        map,
        opacity: isCollision ? 0.35 : rgba[base + 3],
        transparent: isCollision || rgba[base + 3] < 1,
        roughness: isPlane ? 0.95 : 0.55,
        metalness: isPlane ? 0 : 0.15,
        wireframe: isCollision,
      });

      const mesh = new THREE.Mesh(geometry, material);
      mesh.userData.meshId = meshId;
      mesh.castShadow = !isPlane;
      mesh.receiveShadow = true;
      mesh.visible = !isCollision;
      mesh.matrixAutoUpdate = false;
      this.root.add(mesh);
      this.geomObjects.push(mesh);
      if (isCollision) this.collisionObjects.push(mesh);
    }
  }

  setCollisionVisible(visible: boolean): void {
    for (const m of this.collisionObjects) m.visible = visible;
  }

  /**
   * Snapshot the MuJoCo geom poses. Called once per physics step; the pose
   * before it is kept so render() can blend between the two.
   */
  sync(model: MjModel, data: MjData): void {
    const n = model.ngeom;
    if (this.curr.xpos.length !== n * 3) {
      this.curr = { xpos: new Float64Array(n * 3), xmat: new Float64Array(n * 9), follow: new Float64Array(3) };
      this.prev = { xpos: new Float64Array(n * 3), xmat: new Float64Array(n * 9), follow: new Float64Array(3) };
      this.copyPose(this.prev, data);
    } else {
      [this.prev, this.curr] = [this.curr, this.prev];
    }
    this.copyPose(this.curr, data);
  }

  private copyPose(into: Pose, data: MjData): void {
    into.xpos.set(data.geom_xpos.subarray(0, into.xpos.length));
    into.xmat.set(data.geom_xmat.subarray(0, into.xmat.length));
    if (this.followBody !== null) into.follow.set(data.body(this.followBody).xpos.subarray(0, 3));
  }

  /**
   * Write the poses onto the render objects, blended between the last two
   * snapshots by alpha in [0, 1]. Physics ticks at 50 Hz and the display at
   * whatever it likes; without the blend every sixth frame at 60 Hz repeats
   * a pose while the camera keeps gliding, which reads as judder.
   */
  private blend(alpha: number, dt: number): void {
    const a = this.prev, b = this.curr;
    const t = alpha, u = 1 - alpha;
    for (let g = 0; g < this.geomObjects.length; g++) {
      const obj = this.geomObjects[g];
      if (!obj) continue;
      const p = g * 3, m = g * 9;
      const px = u * a.xpos[p] + t * b.xpos[p];
      const py = u * a.xpos[p + 1] + t * b.xpos[p + 1];
      const pz = u * a.xpos[p + 2] + t * b.xpos[p + 2];
      // MuJoCo stores row-major 3x3; three.js Matrix4.set takes row-major too.
      // A linear blend of two rotations 20 ms apart stays orthonormal to well
      // under a pixel.
      this.mat.set(
        u * a.xmat[m + 0] + t * b.xmat[m + 0], u * a.xmat[m + 1] + t * b.xmat[m + 1], u * a.xmat[m + 2] + t * b.xmat[m + 2], px,
        u * a.xmat[m + 3] + t * b.xmat[m + 3], u * a.xmat[m + 4] + t * b.xmat[m + 4], u * a.xmat[m + 5] + t * b.xmat[m + 5], py,
        u * a.xmat[m + 6] + t * b.xmat[m + 6], u * a.xmat[m + 7] + t * b.xmat[m + 7], u * a.xmat[m + 8] + t * b.xmat[m + 8], pz,
        0, 0, 0, 1,
      );
      obj.matrix.copy(this.mat);
      obj.matrixWorldNeedsUpdate = true;
    }

    if (this.followBody !== null) {
      const f = a.follow, g = b.follow;
      // Body pose is in MuJoCo's z-up frame; the root carries the flip.
      this.followTarget
        .set(u * f[0] + t * g[0], u * f[1] + t * g[1], u * f[2] + t * g[2])
        .applyQuaternion(Z_UP);
      // Camera and target move together, so the orbit offset the user set is
      // kept while the robot walks off.
      const k = 1 - Math.exp(-dt / FOLLOW_TAU);
      this.followDelta.copy(this.followTarget).sub(this.controls.target).multiplyScalar(k);
      this.controls.target.add(this.followDelta);
      this.camera.position.add(this.followDelta);
      this.key.target.position.copy(this.controls.target);
      this.key.position.copy(this.controls.target).add(this.keyOffset);
    }
  }

  /** Draw, with the geoms blended alpha of the way from the previous
   *  snapshot to the current one; dt is the wall time since the last draw. */
  render(alpha = 1, dt = 1 / 60): void {
    if (this.curr.xpos.length) this.blend(alpha, dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  private resize(): void {
    const box = this.canvas.parentElement ?? this.canvas;
    const w = box.clientWidth;
    const h = box.clientHeight;
    // A hidden stage (another workspace up front) measures 0x0. Keep the last
    // good size rather than collapsing the buffer and the camera aspect —
    // the observer fires again with real numbers when it comes back.
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  dispose(): void {
    this.resizeObserver.disconnect();
    this.controls.dispose();
    this.disposeObjects();
    this.renderer.dispose();
  }

  private disposeObjects(): void {
    for (const obj of this.geomObjects) {
      if (!obj) continue;
      // Mesh-asset geometries are shared and disposed via meshCache below;
      // primitives are one-per-geom and disposed here.
      if (!this.meshCache.has(obj.userData.meshId ?? -1)) obj.geometry.dispose();
      const mat = obj.material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else {
        if ("map" in mat) (mat.map as THREE.Texture | null)?.dispose();
        mat.dispose();
      }
    }
    for (const geo of this.meshCache.values()) geo.dispose();
    this.meshCache = new Map();
    this.geomObjects.length = 0;
    this.collisionObjects.length = 0;
  }
}
