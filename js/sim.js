// sim.js — ฟิสิกส์และตัวควบคุมการขึ้นบันไดของ DOBOT Atom (พอร์ตจาก atom_stairs_climb.py)
// ไม่ยุ่งกับหน้าเว็บ ใช้ได้ทั้งในเบราว์เซอร์และ Node (สำหรับทดสอบ)

export const LEG_JOINTS = [
  'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
  'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
  'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
  'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
];
// ท่ายืนและ PD gains จาก atom_p3.py / sim2sim.py ของ DOBOT (BSD-3-Clause)
const DEFAULT = [-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0];
const KP = [100, 95, 90, 110, 50, 50, 100, 95, 90, 110, 50, 50];
const KD = [3.5, 2.5, 2.5, 3.5, 1.5, 1.5, 3.5, 2.5, 2.5, 3.5, 1.5, 1.5];
const GAIN_SCALE = 4.0;
const TAU_LIMIT = 200.0;

const STAIR_WIDTH = 1.2, START_X = 0.55, LANDING = 1.0;
const ANKLE_TO_SOLE = 0.066, FOOT_FWD = 0.036, TOE = 0.156, HEEL = 0.084, FOOT_Y = 0.1315;
const PELVIS_ABOVE_ANKLE = 0.80, CLEARANCE = 0.07, MOVE_T = 1.4;
export const CTRL_DT = 0.01;
const HARNESS_KP = 6000, HARNESS_KD = 700, HARNESS_KR = 1500, HARNESS_KRD = 120;

// ---------------------------------------------------------------- บันไดสุ่ม
export function rng(seed) {                 // mulberry32
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function makeStairs(seed, n, hmin, hmax, dmin, dmax) {
  const r = rng(seed);
  const heights = [], depths = [];
  for (let i = 0; i < n; i++) heights.push(hmin + (hmax - hmin) * r());
  for (let i = 0; i < n; i++) depths.push(dmin + (dmax - dmin) * r());
  depths[n - 1] = LANDING;
  return { heights, depths };
}

export function sceneXML(baseXml, heights, depths) {
  let x = START_X, top = 0, geoms = '';
  const col = ['0.78 0.62 0.45 1', '0.70 0.55 0.40 1'];
  heights.forEach((h, i) => {
    top += h;
    const d = depths[i];
    geoms += `<geom name="stair_${i + 1}" type="box" size="${d / 2} ${STAIR_WIDTH / 2} ${top / 2}" ` +
      `pos="${x + d / 2} 0 ${top / 2}" rgba="${col[i % 2]}" friction="1 0.1 0.1" contype="1" conaffinity="1"/>\n`;
    x += d;
  });
  // ทรงกระบอกชนของต้นขากับหน้าแข้งในโมเดลหยาบ งอเข่ามากแล้วชนกันเอง (หุ่นจริงไม่ชน) จึงปิดการชนคู่นี้
  let ex = '';
  for (const s of ['left', 'right']) {
    for (const [a, b] of [[`${s}_hip_roll_link`, `${s}_knee_link`], ['pelvis', `${s}_hip_roll_link`],
      ['pelvis', `${s}_knee_link`], [`${s}_knee_link`, `${s}_ankle_roll_link`]]) {
      ex += `<exclude body1="${a}" body2="${b}"/>\n`;
    }
  }
  const i = baseXml.lastIndexOf('</worldbody>');
  let xml = baseXml.slice(0, i) + geoms + baseXml.slice(i);
  const j = xml.lastIndexOf('</mujoco>');
  xml = xml.slice(0, j) + `<contact>\n${ex}</contact>\n` + xml.slice(j);
  xml = xml.replace(/<site name="(com_marker|joint_marker_\d)"[^>]*\/>/g, '');   // ซ่อนจุดแสดงผลของ DOBOT
  return xml;
}

// ---------------------------------------------------------------- วางแผนก้าวเท้า
const smooth = (s) => { s = Math.min(Math.max(s, 0), 1); return s * s * (3 - 2 * s); };

function treadSpans(heights, depths) {
  let x = START_X, top = 0;
  return heights.map((h, i) => { top += h; const o = [x, x + depths[i], top]; x += depths[i]; return o; });
}

function ankleXOn(x0, x1, landing) {
  if (landing) return x0 + 0.20;
  const want = (x0 + x1) / 2 - FOOT_FWD, hi = x1 - TOE - 0.015, lo = x0 + HEEL + 0.005;
  return lo <= hi ? Math.min(want, hi) : hi;
}

function planFootsteps(heights, depths) {     // step-to: ขวาขึ้นก่อน ซ้ายตามมาขั้นเดียวกัน
  const pre = START_X - 0.03 - TOE;
  const moves = [['R', pre, 0], ['L', pre, 0]];
  const spans = treadSpans(heights, depths);
  spans.forEach(([x0, x1, top], i) => {
    const ax = ankleXOn(x0, x1, i === spans.length - 1);
    moves.push(['R', ax, top], ['L', ax, top]);
  });
  return moves;
}

function buildTimeline(moves, startX) {
  const feet = { L: [startX, 0], R: [startX, 0] };
  const segs = [];
  let t = 1.5;
  for (const [foot, x, top] of moves) {
    const before = { L: [...feet.L], R: [...feet.R] };
    feet[foot] = [x, top];
    segs.push({ t0: t, t1: t + MOVE_T, foot, before, after: { L: [...feet.L], R: [...feet.R] } });
    t += MOVE_T;
  }
  return { segs, tEnd: t + 1.5 };
}

const pelvisOf = (f) => [(f.L[0] + f.R[0]) / 2 + 0.02, 0, Math.min(f.L[1], f.R[1]) + ANKLE_TO_SOLE + PELVIS_ABOVE_ANKLE];
const footPos = (xz, foot) => [xz[0], foot === 'L' ? FOOT_Y : -FOOT_Y, xz[1] + ANKLE_TO_SOLE];
const lerp3 = (a, b, s) => [0, 1, 2].map((i) => (1 - s) * a[i] + s * b[i]);

function targetsAt(t, segs, standZ, startX) {
  const init = { L: [startX, 0], R: [startX, 0] };
  if (t < segs[0].t0) {
    const s = smooth(t / segs[0].t0), p = pelvisOf(init);
    p[2] = (1 - s) * standZ + s * p[2];
    return { L: footPos(init.L, 'L'), R: footPos(init.R, 'R'), p, swing: null };
  }
  const seg = segs.find((g) => t < g.t1);
  if (!seg) {
    const a = segs[segs.length - 1].after;
    return { L: footPos(a.L, 'L'), R: footPos(a.R, 'R'), p: pelvisOf(a), swing: null };
  }
  const s = (t - seg.t0) / (seg.t1 - seg.t0);
  const p = lerp3(pelvisOf(seg.before), pelvisOf(seg.after), smooth(s));
  p[1] = 0.035 * Math.sin(Math.PI * s) * (seg.foot === 'R' ? 1 : -1);
  const pos = { L: footPos(seg.before.L, 'L'), R: footPos(seg.before.R, 'R') };
  const [xa, za] = seg.before[seg.foot], [xb, zb] = seg.after[seg.foot];
  const zc = Math.max(za, zb) + ANKLE_TO_SOLE + CLEARANCE;
  let x, z;
  if (s < 0.3) { const u = smooth(s / 0.3); x = xa; z = (za + ANKLE_TO_SOLE) * (1 - u) + zc * u; }
  else if (s < 0.7) { const u = smooth((s - 0.3) / 0.4); x = xa * (1 - u) + xb * u; z = zc; }
  else { const u = smooth((s - 0.7) / 0.3); x = xb; z = zc * (1 - u) + (zb + ANKLE_TO_SOLE) * u; }
  pos[seg.foot] = [x, seg.foot === 'L' ? FOOT_Y : -FOOT_Y, z];
  return { L: pos.L, R: pos.R, p, swing: seg.foot };
}

function loadShare(t, segs) {
  for (const g of segs) {
    if (g.t0 <= t && t < g.t1) {
      const s = (t - g.t0) / (g.t1 - g.t0);
      const w = s < 0.5 ? 0.5 * (1 - smooth(s / 0.15)) : 0.5 * smooth((s - 0.85) / 0.15);
      return g.foot === 'L' ? [w, 1 - w] : [1 - w, w];
    }
  }
  return [0.5, 0.5];
}

// ---------------------------------------------------------------- คณิตศาสตร์ quaternion / เมทริกซ์เล็ก
const qconj = (q) => [q[0], -q[1], -q[2], -q[3]];
const qmul = (a, b) => [
  a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
  a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
  a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
  a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]];
function qvel(q) {                          // = mju_quat2Vel(q, 1): แกน × มุม
  let [w, x, y, z] = q;
  if (w < 0) { w = -w; x = -x; y = -y; z = -z; }
  const s = Math.hypot(x, y, z);
  if (s < 1e-12) return [0, 0, 0];
  const ang = 2 * Math.atan2(s, w);
  return [x / s * ang, y / s * ang, z / s * ang];
}
function solve(A, b) {                       // Gaussian elimination (n เล็ก)
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

// ---------------------------------------------------------------- ตัวจำลอง
export class StairClimb {
  constructor(mj, model, heights, depths) {
    this.mj = mj; this.m = model; this.d = new mj.MjData(model);
    this.heights = heights; this.depths = depths;
    const id = (type, name) => mj.mj_name2id(model, mj.mjtObj[type].value, name);
    const jids = LEG_JOINTS.map((n) => id('mjOBJ_JOINT', n));
    this.legQ = jids.map((j) => model.jnt_qposadr[j]);
    this.legV = jids.map((j) => model.jnt_dofadr[j]);
    this.range = jids.map((j) => [model.jnt_range[2 * j], model.jnt_range[2 * j + 1]]);
    this.legAct = LEG_JOINTS.map((n) => id('mjOBJ_ACTUATOR', n));
    this.pid = id('mjOBJ_BODY', 'pelvis');
    this.feet = [id('mjOBJ_BODY', 'left_ankle_roll_link'), id('mjOBJ_BODY', 'right_ankle_roll_link')];
    this.weight = model.body_subtreemass[this.pid] * 9.81;
    this.nv = model.nv;
    this.jp = new mj.DoubleBuffer(3 * model.nv);
    this.jr = new mj.DoubleBuffer(3 * model.nv);
    model.opt.timestep = 0.001;
    this.harnessOn = true;
    this.reset();
    this.ready = false;
  }

  reset() {
    const { mj, m, d } = this;
    mj.mj_resetData(m, d);
    const q = d.qpos;
    this.legQ.forEach((a, i) => { q[a] = DEFAULT[i]; });
    mj.mj_forward(m, d);
    this.stats = { fz: 0, side: 0, torque: 0, n: 0, maxFz: 0 };
    this.fallen = false;
  }

  jac(d, body) {
    this.mj.mj_jacBody(this.m, d, this.jp, this.jr, body);
    return [this.jp.GetView(), this.jr.GetView()];
  }

  // วางแผนท่าทางทั้งหมดล่วงหน้า (IK) — ทำทีละช่วง เพื่อให้หน้าเว็บแสดงความคืบหน้าได้
  async plan(onProgress) {
    const { mj, m } = this;
    const startX = this.d.xpos[3 * this.feet[0]];
    const { segs, tEnd } = buildTimeline(planFootsteps(this.heights, this.depths), startX);
    this.tEnd = tEnd;
    const N = Math.ceil(tEnd / CTRL_DT) + 2;
    this.Q = new Float64Array(N * 12); this.P = new Float64Array(N * 3); this.W = new Float64Array(N * 2);
    const di = new mj.MjData(m);
    const q = di.qpos;
    q[3] = 1;
    this.legQ.forEach((a, i) => { q[a] = DEFAULT[i]; });
    for (let k = 0; k < N; k++) {
      const t = k * CTRL_DT;
      const tg = targetsAt(t, segs, 0.955, startX);
      this.ik(di, tg.p, [tg.L, tg.R]);
      const qp = di.qpos;
      this.legQ.forEach((a, i) => { this.Q[12 * k + i] = qp[a]; });
      this.P.set(tg.p, 3 * k);
      this.W.set(loadShare(t, segs), 2 * k);
      if (k % 150 === 0) { onProgress?.(k / N); await new Promise((r) => setTimeout(r, 0)); }
    }
    di.delete();
    this.N = N;
    this.ready = true;
    onProgress?.(1);
  }

  ik(di, pelvis, goals) {
    const { mj, m } = this;
    let q = di.qpos;
    q[0] = pelvis[0]; q[1] = pelvis[1]; q[2] = pelvis[2]; q[3] = 1; q[4] = 0; q[5] = 0; q[6] = 0;
    for (let side = 0; side < 2; side++) {
      const b = this.feet[side], cols = this.legV.slice(6 * side, 6 * side + 6);
      const qa = this.legQ.slice(6 * side, 6 * side + 6), lim = this.range.slice(6 * side, 6 * side + 6);
      for (let it = 0; it < 30; it++) {
        mj.mj_kinematics(m, di);
        mj.mj_comPos(m, di);
        const xp = di.xpos, xq = di.xquat;
        const ep = [0, 1, 2].map((i) => goals[side][i] - xp[3 * b + i]);
        const er = qvel(qconj([xq[4 * b], xq[4 * b + 1], xq[4 * b + 2], xq[4 * b + 3]]));
        if (Math.hypot(...ep) < 1e-4 && Math.hypot(...er) < 1e-3) break;
        const [jp, jr] = this.jac(di, b);
        const J = [];
        for (let r = 0; r < 3; r++) J.push(cols.map((c) => jp[r * this.nv + c]));
        for (let r = 0; r < 3; r++) J.push(cols.map((c) => 0.5 * jr[r * this.nv + c]));
        const e = [...ep, ...er.map((v) => 0.5 * v)];
        const JJt = J.map((ri) => J.map((rj) => ri.reduce((s, v, k) => s + v * rj[k], 0)));
        for (let i = 0; i < 6; i++) JJt[i][i] += 1e-4;
        const y = solve(JJt, e);
        q = di.qpos;
        for (let k = 0; k < 6; k++) {
          let dq = 0;
          for (let r = 0; r < 6; r++) dq += J[r][k] * y[r];
          q[qa[k]] = Math.min(Math.max(q[qa[k]] + dq, lim[k][0]), lim[k][1]);
        }
      }
    }
  }

  step() {
    const { mj, m, d } = this;
    const k = Math.min(Math.floor(d.time / CTRL_DT), this.N - 2);
    const u = (d.time - k * CTRL_DT) / CTRL_DT;
    const at = (A, n, i) => (1 - u) * A[n * k + i] + u * A[n * (k + 1) + i];
    const vel = (A, n, i) => (A[n * (k + 1) + i] - A[n * k + i]) / CTRL_DT;
    const qpos = d.qpos, qvelv = d.qvel;

    // สายแขวนเสมือน
    const xpos = d.xpos, xquat = d.xquat, xmat = d.xmat, pid = this.pid;
    const f = [0, 1, 2].map((i) => HARNESS_KP * (at(this.P, 3, i) - xpos[3 * pid + i]) + HARNESS_KD * (vel(this.P, 3, i) - qvelv[i]));
    const er = qvel(qconj([xquat[4 * pid], xquat[4 * pid + 1], xquat[4 * pid + 2], xquat[4 * pid + 3]]));
    const R = Array.from(xmat.slice(9 * pid, 9 * pid + 9));
    const wl = [qvelv[3], qvelv[4], qvelv[5]];
    const w = [0, 1, 2].map((r) => R[3 * r] * wl[0] + R[3 * r + 1] * wl[1] + R[3 * r + 2] * wl[2]);
    const tq = [0, 1, 2].map((i) => HARNESS_KR * er[i] - HARNESS_KRD * w[i]);
    const xf = d.xfrc_applied;
    for (let i = 0; i < 3; i++) { xf[6 * pid + i] = this.harnessOn ? f[i] : 0; xf[6 * pid + 3 + i] = this.harnessOn ? tq[i] : 0; }

    // มอเตอร์ขา: PD ตามแผน + แรงพยุงน้ำหนักของขาที่ยืน (tau = J^T f)
    const tau = LEG_JOINTS.map((_, i) => GAIN_SCALE * (KP[i] * (at(this.Q, 12, i) - qpos[this.legQ[i]]) +
      KD[i] * (vel(this.Q, 12, i) - qvelv[this.legV[i]])));
    for (let s = 0; s < 2; s++) {
      const share = at(this.W, 2, s);
      const [jp] = this.jac(d, this.feet[s]);
      for (let c = 0; c < 6; c++) tau[6 * s + c] += jp[2 * this.nv + this.legV[6 * s + c]] * (-share * this.weight);
    }
    const ctrl = d.ctrl;
    this.legAct.forEach((a, i) => { ctrl[a] = Math.min(Math.max(tau[i], -TAU_LIMIT), TAU_LIMIT); });
    mj.mj_step(m, d);

    if (this.harnessOn) {
      const st = this.stats;
      st.fz += Math.max(f[2], 0); st.side += Math.hypot(f[0], f[1]); st.torque += Math.hypot(...tq); st.n++;
      st.maxFz = Math.max(st.maxFz, f[2]);
    }
    if (d.xquat[4 * pid] < 0.8) this.fallen = true;   // ลำตัวเอียงเกิน ~74°
  }

  summary() {
    const st = this.stats, n = Math.max(st.n, 1);
    return {
      time: this.d.time, done: this.d.time >= this.tEnd,
      pelvisZ: this.d.xpos[3 * this.pid + 2], top: this.heights.reduce((a, b) => a + b, 0),
      liftPct: st.fz / n / this.weight * 100, sideN: st.side / n, torqueNm: st.torque / n,
      weightKg: this.weight / 9.81, fallen: this.fallen,
    };
  }

  dispose() { this.jp.delete(); this.jr.delete(); this.d.delete(); }
}
