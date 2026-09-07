// Populates public/ with everything the browser app needs at runtime:
//
//   vendor/   MuJoCo + onnxruntime WASM runtimes, copied out of node_modules so
//             they are fetched as plain static files (no bundler in the loop)
//   model/    the Microduck MJCF + the STL meshes it references, from the
//             microduck_rl submodule
//   policies/ the shipped ONNX policies, from the Hugging Face Hub
//
// Everything it writes is gitignored and reproducible: run `npm run
// prepare-assets` (dev/build do it automatically).
import { fileURLToPath } from "node:url";
import fs from "node:fs";
import path from "node:path";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const PUBLIC = path.join(ROOT, "public");
const MJCF_SRC = path.join(ROOT, "microduck_rl/src/mjlab_microduck/robot/microduck");

// Policies live in the standalone Apache-2.0 mirror of the checkpoints that
// ship with the robot; the sandbox Space serves the same files.
const POLICY_BASE =
  "https://huggingface.co/pollen-robotics/microduck-policies/resolve/main";
const POLICIES = [
  "alpha_stand.onnx", // get-up / stand-up, the one the demo screen runs
  "alpha_walking.onnx", // velocity-commanded gait, driven by the remote
];

const log = (...a) => console.log("[assets]", ...a);

function copyIfChanged(src, dst) {
  const s = fs.statSync(src);
  const d = fs.existsSync(dst) ? fs.statSync(dst) : null;
  if (d && d.size === s.size && d.mtimeMs >= s.mtimeMs) return false;
  fs.mkdirSync(path.dirname(dst), { recursive: true });
  fs.copyFileSync(src, dst);
  return true;
}

// ── MuJoCo + onnxruntime WASM ────────────────────────────────────────────
function vendorRuntimes() {
  // "exports" hides package.json, so resolve a mapped entry point instead.
  const mjDir = path.dirname(fileURLToPath(import.meta.resolve("@mujoco/mujoco")));
  let n = 0;
  for (const f of ["mujoco.js", "mujoco.wasm"]) {
    n += copyIfChanged(path.join(mjDir, f), path.join(PUBLIC, "vendor/mujoco", f));
  }
  // onnxruntime-web is imported through the bundler, but it fetches its own
  // .wasm/.mjs sidecars at runtime from env.wasm.wasmPaths — those must be
  // static files. Only the non-threaded SIMD build is used: static hosting
  // sends no COOP/COEP, so SharedArrayBuffer (threads) is unavailable.
  const ortDist = path.dirname(
    fileURLToPath(import.meta.resolve("onnxruntime-web/ort-wasm-simd-threaded.wasm")),
  );
  for (const f of fs.readdirSync(ortDist)) {
    if (/^ort-wasm-simd-threaded\.(wasm|mjs)$/.test(f)) {
      n += copyIfChanged(path.join(ortDist, f), path.join(PUBLIC, "vendor/ort", f));
    }
  }
  log(`runtimes: ${n} file(s) updated`);
}

// ── Microduck MJCF + meshes ──────────────────────────────────────────────
// All three collision variants ship. They declare the same 38 meshes and
// differ only in which geoms use them, so the extra two cost ~60 KB:
//   walk           lightest, what training uses
//   groundcontact  feet + a few shells
//   allcollisions  full shell contact — the standing/sitting policies need it
//                  (a fallen duck rests its trunk on the floor)
const MJCF_FILES = [
  "robot_allcollisions.xml",
  "robot_groundcontact.xml",
  "robot_walk.xml",
];

// Each model also gets a "-nv" (no-visual) twin for training. Visual geoms are
// contype=0 conaffinity=0 and every body carries an explicit <inertial>, so
// dropping them cannot change the dynamics — but it drops the meshes only they
// referenced, and with them MuJoCo's convex-hull work at compile time. On
// robot_walk that is 34 of 38 meshes, which is most of the ~538 MB a worker
// spends compiling. Training never renders, so it always wants these.
const NO_VISUAL_SUFFIX = "-nv";

/** Meshes referenced by any <geom mesh="..."> still present in the document. */
function referencedMeshes(xml) {
  const used = new Set();
  for (const geom of xml.match(/<geom\b[^>]*>/g) ?? []) {
    const m = /\bmesh="([^"]+)"/.exec(geom);
    if (m) used.add(m[1]);
  }
  return used;
}

/** name -> file for every <mesh> in <asset>. */
function declaredMeshes(xml) {
  const byName = new Map();
  for (const mesh of xml.match(/<mesh\b[^>]*\/>/g) ?? []) {
    const file = /\bfile="([^"]+)"/.exec(mesh)?.[1];
    if (!file) continue;
    const name = /\bname="([^"]+)"/.exec(mesh)?.[1] ?? file.replace(/\.stl$/i, "");
    byName.set(name, { file, tag: mesh });
  }
  return byName;
}

/** Drop visual geoms, then every <mesh> nothing references any more. */
function stripVisuals(xml) {
  const stripped = xml.replace(/[ \t]*<geom\b[^>]*\bclass="visual"[^>]*>\s*\n?/g, "");
  const used = referencedMeshes(stripped);
  let out = stripped;
  for (const [name, { tag }] of declaredMeshes(stripped)) {
    if (!used.has(name)) out = out.replace(tag + "\n", "").replace(tag, "");
  }
  return out;
}

function meshFilesFor(xml) {
  const declared = declaredMeshes(xml);
  return [...referencedMeshes(xml)]
    .map((name) => declared.get(name)?.file)
    .filter((f) => f !== undefined)
    .sort();
}

function vendorModel() {
  if (!fs.existsSync(MJCF_SRC)) {
    throw new Error(
      `microduck_rl submodule not checked out (${MJCF_SRC} missing).\n` +
        "Run: git submodule update --init --recursive",
    );
  }
  const dst = path.join(PUBLIC, "model/microduck");
  fs.mkdirSync(dst, { recursive: true });
  let n = 0;
  const models = [];
  const allMeshes = new Set();

  for (const f of MJCF_FILES) {
    const src = path.join(MJCF_SRC, f);
    const xml = fs.readFileSync(src, "utf8");
    n += copyIfChanged(src, path.join(dst, f));
    models.push({ file: f, meshes: meshFilesFor(xml) });

    const nvFile = f.replace(/\.xml$/, `${NO_VISUAL_SUFFIX}.xml`);
    const nvXml = stripVisuals(xml);
    const nvMeshes = meshFilesFor(nvXml);
    if (nvMeshes.length === 0) throw new Error(`${f}: stripping visuals left no meshes`);
    fs.writeFileSync(path.join(dst, nvFile), nvXml);
    models.push({ file: nvFile, meshes: nvMeshes });
    n++;

    for (const m of [...models.at(-1).meshes, ...models.at(-2).meshes]) allMeshes.add(m);
  }

  for (const mesh of allMeshes) {
    n += copyIfChanged(path.join(MJCF_SRC, "assets", mesh), path.join(dst, "assets", mesh));
  }

  // The app fetches this instead of parsing every MJCF in the browser just to
  // learn which meshes it needs.
  const manifest = { models, meshes: [...allMeshes].sort() };
  fs.writeFileSync(path.join(dst, "manifest.json"), JSON.stringify(manifest, null, 2));
  const summary = models.map((m) => `${m.file} (${m.meshes.length})`).join(", ");
  log(`model: ${summary}; ${n} file(s) updated`);
  return { dst, models };
}

// ── Verify the stripped models ───────────────────────────────────────────
// Dropping visual geoms MUST NOT change the dynamics. Rather than trust the
// regex, compile both twins with the real MuJoCo and compare the numbers that
// would break a policy: degrees of freedom, actuators, and the mass matrix.
async function verifyModels({ dst, models }) {
  const { default: loadMujoco } = await import("@mujoco/mujoco");
  const mujoco = await loadMujoco();
  const vfs = new mujoco.MjVFS();
  for (const m of models) {
    vfs.addBuffer(m.file, new Uint8Array(fs.readFileSync(path.join(dst, m.file))));
  }
  for (const mesh of fs.readdirSync(path.join(dst, "assets"))) {
    vfs.addBuffer(`assets/${mesh}`, new Uint8Array(fs.readFileSync(path.join(dst, "assets", mesh))));
  }

  const compile = (file) =>
    mujoco.MjModel.from_xml_string(
      `<mujoco><include file="${file}"/><option timestep="0.005"/>` +
        `<worldbody><geom name="floor" type="plane" size="0 0 .05"/></worldbody></mujoco>`,
      vfs,
    );

  for (const { file } of models) {
    if (!file.includes(NO_VISUAL_SUFFIX)) continue;
    const full = compile(file.replace(`${NO_VISUAL_SUFFIX}.xml`, ".xml"));
    const nv = compile(file);
    const same = ["nq", "nv", "nu", "njnt", "nbody"].filter((k) => full[k] !== nv[k]);
    if (same.length) {
      throw new Error(`${file}: stripping visuals changed ${same.join(", ")}`);
    }
    // Body masses catch a geom that was contributing inertia despite the class.
    const mass = (m) => Array.from(m.body_mass).reduce((a, b) => a + b, 0);
    const drift = Math.abs(mass(full) - mass(nv));
    if (drift > 1e-9) throw new Error(`${file}: total mass moved by ${drift}`);
    log(`verify: ${file} matches its source (nq=${nv.nq} nu=${nv.nu}, ${nv.ngeom} geoms vs ${full.ngeom})`);
  }
}

// ── ONNX policies ────────────────────────────────────────────────────────
async function fetchPolicies() {
  const dst = path.join(PUBLIC, "policies");
  fs.mkdirSync(dst, { recursive: true });
  for (const name of POLICIES) {
    const out = path.join(dst, name);
    if (fs.existsSync(out) && fs.statSync(out).size > 0) {
      log(`policy: ${name} (cached)`);
      continue;
    }
    const url = `${POLICY_BASE}/${name}`;
    log(`policy: fetching ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`${url} -> HTTP ${res.status}`);
    fs.writeFileSync(out, Buffer.from(await res.arrayBuffer()));
  }
}

vendorRuntimes();
const model = vendorModel();
await verifyModels(model);
await fetchPolicies();
log("done");
