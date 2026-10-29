// main.js — หน้าเว็บ: โหลดโมเดล, วาด 3D ด้วย three.js, ปุ่มควบคุม
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import loadMujoco from '@mujoco/mujoco';
import { makeStairs, sceneXML, StairClimb } from './sim.js';

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// โมเดลหุ่นโหลดตรงจาก repo ของ DOBOT (ไม่ได้เก็บไว้ในเว็บนี้)
const SOURCES = params.get('assets') ? [params.get('assets')] : [
  'https://raw.githubusercontent.com/embodied-dobot/atom-locomotion-training/main/resources/robots/ATOM',
  'https://cdn.jsdelivr.net/gh/embodied-dobot/atom-locomotion-training@main/resources/robots/ATOM',
];

let mj, sim = null, model = null, running = true, localFiles = null;
const loaded = {};          // รุ่นหุ่นที่เขียนลงระบบไฟล์ของ MuJoCo แล้ว -> XML ต้นฉบับ
const geomCache = new Map(); // mesh id -> BufferGeometry

// ---------------------------------------------------------------- 3D
const view = $('view');
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
view.appendChild(renderer.domElement);
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.05, 100);
camera.up.set(0, 0, 1);
camera.position.set(-2.0, -4.4, 2.1);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0.6, 0, 0.8);
controls.enableDamping = true;

const css = getComputedStyle(document.documentElement);
const dark = matchMedia('(prefers-color-scheme: dark)').matches;
scene.background = new THREE.Color(css.getPropertyValue('--sky-bottom').trim() || '#f3efe8');
scene.fog = new THREE.Fog(scene.background, 9, 26);
scene.add(new THREE.HemisphereLight(dark ? 0x8899aa : 0xffffff, dark ? 0x222018 : 0x8a7f70, dark ? 1.1 : 1.4));
const sun = new THREE.DirectionalLight(0xffffff, dark ? 1.6 : 2.2);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
Object.assign(sun.shadow.camera, { left: -3, right: 3, top: 3, bottom: -3, near: 0.5, far: 15 });
scene.add(sun, sun.target);

{ // พื้นลายตาราง
  const c = document.createElement('canvas'); c.width = c.height = 128;
  const g = c.getContext('2d');
  const a = dark ? '#2b2822' : '#d8d1c4', b = dark ? '#24211c' : '#cfc7b8';
  g.fillStyle = a; g.fillRect(0, 0, 128, 128); g.fillStyle = b; g.fillRect(0, 0, 64, 64); g.fillRect(64, 64, 64, 64);
  const tex = new THREE.CanvasTexture(c);
  tex.wrapS = tex.wrapT = THREE.RepeatWrapping; tex.repeat.set(40, 40); tex.colorSpace = THREE.SRGBColorSpace;
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(40, 40), new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95 }));
  floor.receiveShadow = true;
  scene.add(floor);
}

const bodyGroup = new THREE.Group();
scene.add(bodyGroup);
let geomObjs = [];
const robotMat = new THREE.MeshStandardMaterial({ color: dark ? 0xc9ccd1 : 0xd7d9dd, metalness: 0.35, roughness: 0.45, flatShading: true });

function buildMeshes() {
  bodyGroup.clear();
  geomObjs = [];
  const m = model;
  for (let g = 0; g < m.ngeom; g++) {
    const type = m.geom_type[g], group = m.geom_group[g];
    if (group >= 2) continue;                          // พื้นและรูปทรงชน (ซ่อน)
    let geo, mat;
    if (type === 7) {                                  // mesh
      const id = m.geom_dataid[g];
      if (!geomCache.has(id)) {
        const va = m.mesh_vertadr[id], vn = m.mesh_vertnum[id], fa = m.mesh_faceadr[id], fn = m.mesh_facenum[id];
        const b = new THREE.BufferGeometry();
        b.setAttribute('position', new THREE.BufferAttribute(new Float32Array(m.mesh_vert.slice(3 * va, 3 * (va + vn))), 3));
        b.setIndex(new THREE.BufferAttribute(new Uint32Array(m.mesh_face.slice(3 * fa, 3 * (fa + fn))), 1));
        b.computeVertexNormals();
        geomCache.set(id, b);
      }
      geo = geomCache.get(id); mat = robotMat;
    } else if (type === 6) {                           // box (บันได)
      const s = m.geom_size;
      geo = new THREE.BoxGeometry(2 * s[3 * g], 2 * s[3 * g + 1], 2 * s[3 * g + 2]);
      const r = m.geom_rgba;
      mat = new THREE.MeshStandardMaterial({ color: new THREE.Color(r[4 * g], r[4 * g + 1], r[4 * g + 2]), roughness: 0.85 });
    } else continue;
    const o = new THREE.Mesh(geo, mat);
    o.castShadow = o.receiveShadow = true;
    o.matrixAutoUpdate = false;
    bodyGroup.add(o);
    geomObjs.push([g, o]);
  }
}

const M4 = new THREE.Matrix4();
function syncMeshes() {
  const p = sim.d.geom_xpos, R = sim.d.geom_xmat;
  for (const [g, o] of geomObjs) {
    const r = 9 * g, t = 3 * g;
    M4.set(R[r], R[r + 1], R[r + 2], p[t], R[r + 3], R[r + 4], R[r + 5], p[t + 1], R[r + 6], R[r + 7], R[r + 8], p[t + 2], 0, 0, 0, 1);
    o.matrix.copy(M4);
  }
}

function resize() {
  const w = innerWidth, h = innerHeight;
  renderer.setSize(w, h); camera.aspect = w / h;
  camera.fov = w / h < 1 ? 62 : 40;               // จอแนวตั้ง (มือถือ) มุมกว้างขึ้น ให้เห็นทั้งตัว
  camera.updateProjectionMatrix();
}
addEventListener('resize', resize); resize();

// ---------------------------------------------------------------- โหลดโมเดลหุ่น
const setLoad = (frac, msg) => { $('bar').style.width = `${Math.round(frac * 100)}%`; if (msg) $('loadmsg').textContent = msg; };

async function fetchAsset(rel) {
  if (localFiles) {
    const f = localFiles.get(rel);
    if (!f) throw new Error(`ไม่พบไฟล์ ${rel} ในโฟลเดอร์ที่เลือก`);
    return new Uint8Array(await f.arrayBuffer());
  }
  let err;
  for (const base of SOURCES) {
    try {
      const r = await fetch(`${base}/${rel}`);
      if (r.ok) return new Uint8Array(await r.arrayBuffer());
      err = new Error(`${r.status} ${base}/${rel}`);
    } catch (e) { err = e; }
  }
  throw err;
}

async function ensureRobot(variant) {
  if (loaded[variant]) return loaded[variant];
  showOverlay(true);
  setLoad(0, `ดาวน์โหลดโมเดล Atom ${variant.toUpperCase()} จาก GitHub ของ DOBOT…`);
  const xml = new TextDecoder().decode(await fetchAsset(`mjcf/atom_${variant}_12dof.xml`));
  const meshDir = (xml.match(/meshdir="\.\.\/([^"/]+)\/?"/) || [])[1] || `meshes_${variant}`;
  const files = [...new Set([...xml.matchAll(/<mesh[^>]*file="([^"]+)"/g)].map((x) => x[1]))];
  for (const p of ['/atom', '/atom/mjcf', `/atom/${meshDir}`]) { try { mj.FS.mkdir(p); } catch { /* มีแล้ว */ } }
  let done = 0;
  const queue = [...files];
  const worker = async () => {
    while (queue.length) {
      const f = queue.shift();
      mj.FS.writeFile(`/atom/${meshDir}/${f}`, await fetchAsset(`${meshDir}/${f}`));
      setLoad(++done / files.length, `ดาวน์โหลดชิ้นส่วน 3D ${done}/${files.length}`);
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  loaded[variant] = xml;
  return xml;
}

function showOverlay(on) { $('overlay').classList.toggle('hide', !on); }

// ---------------------------------------------------------------- สร้างฉากใหม่
let building = false;
async function rebuild() {
  if (building) return;
  building = true;
  const variant = $('model').value;
  const cm = (id) => Number($(id).value) / 100;
  let hmin = cm('hmin'), hmax = cm('hmax'), dmin = cm('dmin'), dmax = cm('dmax');
  if (hmin > hmax) [hmin, hmax] = [hmax, hmin];
  if (dmin > dmax) [dmin, dmax] = [dmax, dmin];
  const n = Math.min(15, Math.max(3, Number($('steps').value) || 10));
  const seed = Number($('seed').value) || 0;
  try {
    const base = await ensureRobot(variant);
    const { heights, depths } = makeStairs(seed, n, hmin, hmax, dmin, dmax);
    mj.FS.writeFile('/atom/mjcf/scene.xml', sceneXML(base, heights, depths));
    if (sim) { sim.dispose(); }
    if (model && model !== null) { const old = model; model = null; geomCache.clear(); old.delete(); }
    model = mj.MjModel.from_xml_path('/atom/mjcf/scene.xml');
    sim = new StairClimb(mj, model, heights, depths);
    sim.harnessOn = $('harness').checked;
    buildMeshes();
    syncMeshes();
    showOverlay(true);
    setLoad(0, 'วางแผนก้าวเท้าและคำนวณมุมข้อต่อ (IK)…');
    await sim.plan((f) => setLoad(f));
    fillTable(heights, depths);
    showOverlay(false);
    $('status').textContent = ''; $('status').className = '';
    simClock = 0;
  } catch (e) {
    console.error(e);
    $('loadmsg').textContent = `เกิดข้อผิดพลาด: ${e.message}`;
    $('fallback').style.display = 'block';
    showOverlay(true);
  }
  building = false;
}

function fillTable(h, d) {
  let top = 0;
  const rows = h.map((x, i) => {
    top += x;
    const land = i === h.length - 1 ? ' (ชานพัก)' : '';
    return `<tr><td>${i + 1}${land}</td><td>${(x * 100).toFixed(1)}</td><td>${(d[i] * 100).toFixed(1)}</td><td>${(top * 100).toFixed(1)}</td></tr>`;
  }).join('');
  $('tbl').innerHTML = `<tr><th>ขั้น</th><th>สูง ซม.</th><th>ลึก ซม.</th><th>สูงสะสม</th></tr>${rows}`;
}

// ---------------------------------------------------------------- ลูปหลัก
let simClock = 0, last = performance.now(), finishedAt = null;
function frame(now) {
  const dt = Math.min((now - last) / 1000, 0.05);
  last = now;
  if (sim?.ready && running && !building) {
    simClock = Math.max(simClock, sim.d.time - 0.05) + dt * Number($('speed').value);
    const t0 = performance.now();
    while (sim.d.time < simClock && sim.d.time < sim.tEnd + 0.5) {
      sim.step();
      if (performance.now() - t0 > 30) { simClock = sim.d.time; break; }   // เครื่องช้า: เล่นแบบสโลว์แทน
    }
    syncMeshes();
    updateStats();
    if (sim.d.time >= sim.tEnd && finishedAt === null) finishedAt = now;
    if (finishedAt !== null && $('loop').checked && now - finishedAt > 2500) { restart(); }
  }
  if (sim && $('follow').checked) {
    const pid = sim.pid, x = sim.d.xpos[3 * pid], z = sim.d.xpos[3 * pid + 2];
    const goal = new THREE.Vector3(x + 0.3, 0, z - 0.1);
    const delta = goal.clone().sub(controls.target).multiplyScalar(0.06);
    controls.target.add(delta); camera.position.add(delta);
    sun.position.set(controls.target.x - 2, controls.target.y - 3, controls.target.z + 6);
    sun.target.position.copy(controls.target);
  }
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(frame);
}

function updateStats() {
  const s = sim.summary();
  $('sTime').textContent = `${s.time.toFixed(1)} s`;
  const lowest = Math.min(sim.d.xpos[3 * sim.feet[0] + 2], sim.d.xpos[3 * sim.feet[1] + 2]) - 0.066;
  let top = 0, k = 0;
  for (const h of sim.heights) { if (lowest > top + h - 0.02) { top += h; k++; } else break; }
  $('sStep').textContent = `${k} / ${sim.heights.length}`;
  $('sLift').textContent = sim.harnessOn ? `${s.liftPct.toFixed(0)}%` : 'ปิดอยู่';
  $('sBal').textContent = sim.harnessOn ? `${s.sideN.toFixed(0)} · ${s.torqueNm.toFixed(0)}` : 'ปิดอยู่';
  const st = $('status');
  if (s.fallen) { st.textContent = 'หุ่นล้ม — ไม่มีสายแขวนช่วยทรงตัว'; st.className = 'bad'; }
  else if (s.done) {
    const ok = s.pelvisZ > s.top + 0.5;
    st.textContent = ok ? `ถึงชานพักด้านบนแล้ว ✓  สูง ${(s.top * 100).toFixed(0)} ซม.` : 'ไม่ถึงบนสุด';
    st.className = ok ? 'good' : 'bad';
  } else { st.textContent = ''; st.className = ''; }
}

function restart() {
  if (!sim) return;
  sim.reset();
  sim.harnessOn = $('harness').checked;
  simClock = 0; finishedAt = null;
  $('status').textContent = ''; $('status').className = '';
}

// ---------------------------------------------------------------- ปุ่ม
$('play').onclick = () => { running = !running; $('play').textContent = running ? '⏸ หยุด' : '▶ เล่น'; };
$('restart').onclick = restart;
$('dice').onclick = () => { $('seed').value = Math.floor(Math.random() * 100000); finishedAt = null; rebuild(); };
$('apply').onclick = () => { finishedAt = null; rebuild(); };
$('model').onchange = () => { finishedAt = null; rebuild(); };
$('seed').onchange = () => { finishedAt = null; rebuild(); };
$('harness').onchange = () => { if (sim) sim.harnessOn = $('harness').checked; };
if (matchMedia('(max-width: 640px)').matches) { $('panel').classList.add('collapsed'); $('toggle').textContent = '▴'; }
$('toggle').onclick = () => { $('panel').classList.toggle('collapsed'); $('toggle').textContent = $('panel').classList.contains('collapsed') ? '▴' : '▾'; };
$('folder').onchange = (e) => {
  localFiles = new Map();
  for (const f of e.target.files) {
    const parts = f.webkitRelativePath.split('/');
    const i = parts.findIndex((p) => p === 'mjcf' || p.startsWith('meshes_'));
    if (i >= 0) localFiles.set(parts.slice(i).join('/'), f);
  }
  $('fallback').style.display = 'none';
  rebuild();
};

// สำหรับลองเล่นใน DevTools: atom.sim
window.atom = { get sim() { return sim; } };

// ---------------------------------------------------------------- เริ่ม
(async () => {
  try {
    mj = await loadMujoco();
  } catch (e) {
    $('loadmsg').textContent = `โหลด MuJoCo ไม่สำเร็จ: ${e.message}`;
    return;
  }
  if (params.get('seed')) $('seed').value = params.get('seed');
  await rebuild();
  requestAnimationFrame(frame);
})();
