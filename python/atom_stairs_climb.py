"""
atom_stairs_climb.py — หุ่น DOBOT Atom ก้าวขึ้นบันไดสุ่ม 10 ขั้น ใน MuJoCo (แบบมี "สายแขวนเสมือน" ช่วย)

⚠ นี่คือเดโมแบบมีตัวช่วย ไม่ใช่การทรงตัวจริง:
   - ตำแหน่งวางเท้าวางแผนจากขนาดบันไดที่รู้ล่วงหน้า (ยังไม่ได้ใช้กล้อง)
   - มุมข้อต่อขาคำนวณด้วย Inverse Kinematics แล้วให้มอเตอร์ขา (PD + แรงพยุงน้ำหนัก) ไล่ตาม
   - "สายแขวนเสมือน" ดึงสะโพกเข้าหาแผนและบังคับลำตัวให้ตั้งตรง เหมือนสายกันล้มในห้องแล็บ
     สคริปต์รายงานว่าสายแขวนออกแรงช่วยไปกี่ % ของน้ำหนักตัว — ถ้าถอดออก หุ่นจะล้ม
   การทรงตัวจริงต้องได้จาก RL policy

ต้องวางไว้โฟลเดอร์เดียวกับ atom_stand_demo.py และ atom_stairs_demo.py

วิธีใช้:
    python atom_stairs_climb.py --repo .\\atom-locomotion-training-main              (เปิดหน้าต่าง 3D)
    python atom_stairs_climb.py --repo ... --seed 7                                   (บันไดชุดอื่น)
    python atom_stairs_climb.py --repo ... --speed 0.5                                (เล่นช้าลงครึ่งหนึ่ง)
    python atom_stairs_climb.py --repo ... --record climb.mp4                         (บันทึกวิดีโอ ต้องมี ffmpeg)
"""
import argparse
import math
import os
import subprocess
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atom_stand_demo as base  # noqa: E402
import atom_stairs_demo as stairs  # noqa: E402

ANKLE_TO_SOLE = 0.066      # m ระยะจากข้อเท้า (ankle_roll_link) ถึงพื้นรองเท้า
FOOT_FWD = 0.036           # m กึ่งกลางเท้าอยู่หน้าข้อเท้า
TOE, HEEL = 0.156, 0.084   # m ระยะปลายเท้า / ส้นเท้า จากข้อเท้า
FOOT_Y = 0.1315
PELVIS_ABOVE_ANKLE = 0.80  # m ความสูงสะโพกเหนือข้อเท้าของเท้าที่อยู่ขั้นต่ำกว่า
CLEARANCE = 0.07           # m ยกพื้นรองเท้าพ้นขั้นที่สูงกว่า
MOVE_T = 1.4               # s เวลาต่อหนึ่งก้าว
CTRL_DT = 0.01             # s ความถี่ของแผนท่าทาง (100 Hz)

HARNESS_KP, HARNESS_KD = 6000.0, 700.0     # N/m, N·s/m   ดึงสะโพกเข้าหาตำแหน่งในแผน
HARNESS_KR, HARNESS_KRD = 1500.0, 120.0    # N·m/rad      บังคับลำตัวให้ตั้งตรง


def smooth(s):
    s = min(max(s, 0.0), 1.0)
    return s * s * (3 - 2 * s)


# ---------------------------------------------------------------- ฉาก
def build_scene(model_path, heights, depths):
    spec = mujoco.MjSpec.from_file(model_path)
    stairs.add_stairs(spec, heights, depths)
    # ทรงกระบอกชนของต้นขากับหน้าแข้งในโมเดลเป็นรูปทรงหยาบ ถ้างอเข่ามากจะชนกันเอง (หุ่นจริงไม่ชน)
    for s in ("left", "right"):
        for a, b in ((f"{s}_hip_roll_link", f"{s}_knee_link"), ("pelvis", f"{s}_hip_roll_link"),
                     ("pelvis", f"{s}_knee_link"), (f"{s}_knee_link", f"{s}_ankle_roll_link")):
            spec.add_exclude(bodyname1=a, bodyname2=b)
    m = spec.compile()
    m.opt.timestep = 0.001
    d = mujoco.MjData(m)
    return m, d


# ---------------------------------------------------------------- วางแผนก้าวเท้า
def tread_spans(heights, depths):
    x, top, out = stairs.START_X, 0.0, []
    for h, d in zip(heights, depths):
        top += h
        out.append((x, x + d, top))
        x += d
    return out


def ankle_x_on(x0, x1, landing=False):
    """ตำแหน่งข้อเท้าให้เท้าอยู่กลางขั้น โดยปลายเท้าไม่ชนขั้นถัดไป"""
    if landing:
        return x0 + 0.20
    want = (x0 + x1) / 2 - FOOT_FWD
    hi, lo = x1 - TOE - 0.015, x0 + HEEL + 0.005
    return min(want, hi) if lo <= hi else hi


def plan_footsteps(heights, depths):
    """ก้าวแบบ step-to: เท้าขวาขึ้นก่อน เท้าซ้ายตามมาขั้นเดียวกัน (มั่นคงที่สุดกับบันไดไม่สม่ำเสมอ)"""
    moves = []
    pre = stairs.START_X - 0.03 - TOE                       # เดินเข้าไปชิดขั้นแรก
    moves += [("R", pre, 0.0), ("L", pre, 0.0)]
    spans = tread_spans(heights, depths)
    for i, (x0, x1, top) in enumerate(spans):
        ax = ankle_x_on(x0, x1, landing=(i == len(spans) - 1))
        moves += [("R", ax, top), ("L", ax, top)]
    return moves


def build_timeline(moves, start_ankle_x):
    feet = {"L": [start_ankle_x, 0.0], "R": [start_ankle_x, 0.0]}
    segs, t = [], 1.5                                       # 1.5 วินาทีแรก: ย่อตัวจากท่ายืน
    for foot, x, top in moves:
        before = {k: v[:] for k, v in feet.items()}
        feet[foot] = [x, top]
        segs.append((t, t + MOVE_T, foot, before, {k: v[:] for k, v in feet.items()}))
        t += MOVE_T
    return segs, t + 1.5


def pelvis_of(feet):
    x = (feet["L"][0] + feet["R"][0]) / 2 + 0.02
    z = min(feet["L"][1], feet["R"][1]) + ANKLE_TO_SOLE + PELVIS_ABOVE_ANKLE
    return np.array([x, 0.0, z])


def foot_pos(xz, foot):
    return np.array([xz[0], FOOT_Y if foot == "L" else -FOOT_Y, xz[1] + ANKLE_TO_SOLE])


def targets_at(t, segs, stand_pelvis_z, start_ankle_x):
    """เวลา t -> (ข้อเท้าซ้าย, ข้อเท้าขวา, สะโพก, เท้าที่กำลังก้าว)"""
    init = {"L": [start_ankle_x, 0.0], "R": [start_ankle_x, 0.0]}
    if t < segs[0][0]:
        s = smooth(t / segs[0][0])
        p = pelvis_of(init)
        p[2] = (1 - s) * stand_pelvis_z + s * p[2]
        return foot_pos(init["L"], "L"), foot_pos(init["R"], "R"), p, None
    seg = next((sg for sg in segs if t < sg[1]), None)
    if seg is None:
        after = segs[-1][4]
        return foot_pos(after["L"], "L"), foot_pos(after["R"], "R"), pelvis_of(after), None
    t0, t1, foot, before, after = seg
    s = (t - t0) / (t1 - t0)
    p = (1 - smooth(s)) * pelvis_of(before) + smooth(s) * pelvis_of(after)
    p[1] = 0.035 * math.sin(math.pi * s) * (1 if foot == "R" else -1)   # เอียงสะโพกไปฝั่งขาที่ยืน
    pos = {k: foot_pos(before[k], k) for k in ("L", "R")}
    (xa, za), (xb, zb) = before[foot], after[foot]
    z_clear = max(za, zb) + ANKLE_TO_SOLE + CLEARANCE
    if s < 0.3:                                # 1) ยกเท้าขึ้นตรง ๆ
        u = smooth(s / 0.3)
        x, z = xa, (za + ANKLE_TO_SOLE) * (1 - u) + z_clear * u
    elif s < 0.7:                              # 2) ยื่นเท้าไปข้างหน้าเหนือขอบขั้น
        u = smooth((s - 0.3) / 0.4)
        x, z = xa * (1 - u) + xb * u, z_clear
    else:                                      # 3) วางเท้าลงบนขั้น
        u = smooth((s - 0.7) / 0.3)
        x, z = xb, z_clear * (1 - u) + (zb + ANKLE_TO_SOLE) * u
    pos[foot] = np.array([x, FOOT_Y if foot == "L" else -FOOT_Y, z])
    return pos["L"], pos["R"], p, foot


def load_share(t, segs):
    """สัดส่วนน้ำหนักที่ขาซ้าย/ขวาต้องรับ: ขาที่กำลังก้าว = 0, ยืนสองขา = ครึ่งต่อครึ่ง"""
    for t0, t1, foot, _, _ in segs:
        if t0 <= t < t1:
            s = (t - t0) / (t1 - t0)
            w = 0.5 * (1 - smooth(s / 0.15)) if s < 0.5 else 0.5 * smooth((s - 0.85) / 0.15)
            return (w, 1 - w) if foot == "L" else (1 - w, w)
    return 0.5, 0.5


# ---------------------------------------------------------------- Inverse Kinematics
class LegIK:
    def __init__(self, m):
        self.m, self.d = m, mujoco.MjData(m)
        self.feet = {"L": m.body("left_ankle_roll_link").id, "R": m.body("right_ankle_roll_link").id}
        self.qadr, self.vadr, self.rng = {}, {}, {}
        for s, o in (("L", 0), ("R", 6)):
            js = [m.joint(n).id for n in base.LEG_JOINTS[o:o + 6]]
            self.qadr[s] = [m.jnt_qposadr[j] for j in js]
            self.vadr[s] = [m.jnt_dofadr[j] for j in js]
            self.rng[s] = np.array([m.jnt_range[j] for j in js])
        self.d.qpos[3] = 1.0
        self.d.qpos[[*self.qadr["L"], *self.qadr["R"]]] = base.DEFAULT
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

    def solve(self, pelvis, targets, iters=30):
        d = self.d
        d.qpos[0:3], d.qpos[3:7] = pelvis, [1, 0, 0, 0]
        worst = 0.0
        qc, qe, er = np.zeros(4), np.zeros(4), np.zeros(3)
        for side, goal in targets.items():
            b, qa, va, lim = self.feet[side], self.qadr[side], self.vadr[side], self.rng[side]
            for _ in range(iters):
                mujoco.mj_kinematics(self.m, d)
                mujoco.mj_comPos(self.m, d)
                ep = goal - d.xpos[b]
                mujoco.mju_negQuat(qc, d.xquat[b])
                mujoco.mju_mulQuat(qe, np.array([1.0, 0, 0, 0]), qc)   # เท้าขนานพื้น หันไปข้างหน้า
                mujoco.mju_quat2Vel(er, qe, 1.0)
                if np.linalg.norm(ep) < 1e-4 and np.linalg.norm(er) < 1e-3:
                    break
                mujoco.mj_jacBody(self.m, d, self.jp, self.jr, b)
                J = np.vstack([self.jp[:, va], 0.5 * self.jr[:, va]])
                e = np.concatenate([ep, 0.5 * er])
                dq = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(6), e)
                d.qpos[qa] = np.clip(d.qpos[qa] + dq, lim[:, 0], lim[:, 1])
            mujoco.mj_kinematics(self.m, d)
            worst = max(worst, float(np.linalg.norm(goal - d.xpos[b])))
        return np.concatenate([d.qpos[self.qadr["L"]], d.qpos[self.qadr["R"]]]), worst


# ---------------------------------------------------------------- ตัวควบคุม
class Climber:
    def __init__(self, m, d, heights, depths, verbose=True):
        self.m, self.d, self.heights = m, d, heights
        self.leg_q = [m.jnt_qposadr[m.joint(n).id] for n in base.LEG_JOINTS]
        self.leg_v = [m.jnt_dofadr[m.joint(n).id] for n in base.LEG_JOINTS]
        self.leg_act = [m.actuator(n).id for n in base.LEG_JOINTS]
        self.pid = m.body("pelvis").id
        self.feet = (m.body("left_ankle_roll_link").id, m.body("right_ankle_roll_link").id)
        self.weight = float(m.body_subtreemass[self.pid]) * 9.81
        self.jp, self.jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))

        self.reset()
        start_x = float(d.xpos[self.feet[0]][0])
        segs, self.t_end = build_timeline(plan_footsteps(heights, depths), start_x)
        ik = LegIK(m)
        self.times = np.arange(0, self.t_end + 2 * CTRL_DT, CTRL_DT)
        if verbose:
            print(f"คำนวณ IK {len(self.times)} ท่า ({len(segs)} ก้าว, {self.t_end:.0f} วินาที) ...", end=" ", flush=True)
        Q, P, WS, worst = [], [], [], 0.0
        for t in self.times:
            lf, rf, p, _ = targets_at(t, segs, 0.955, start_x)
            q, err = ik.solve(p, {"L": lf, "R": rf})
            Q.append(q)
            P.append(p)
            WS.append(load_share(t, segs))
            worst = max(worst, err)
        self.Q, self.P, self.WS = np.array(Q), np.array(P), np.array(WS)
        self.Qd = np.gradient(self.Q, CTRL_DT, axis=0)
        self.Pd = np.gradient(self.P, CTRL_DT, axis=0)
        if verbose:
            print(f"เสร็จ (คลาดเคลื่อนสูงสุด {worst * 1000:.1f} มม.)")

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.leg_q] = base.DEFAULT
        mujoco.mj_forward(self.m, self.d)
        self.fz, self.side, self.torque = [], [], []

    def harness(self, p_des, v_des):
        d, pid = self.d, self.pid
        f = HARNESS_KP * (p_des - d.xpos[pid]) + HARNESS_KD * (v_des - d.qvel[0:3])
        qc, qe, er = np.zeros(4), np.zeros(4), np.zeros(3)
        mujoco.mju_negQuat(qc, d.xquat[pid])
        mujoco.mju_mulQuat(qe, np.array([1.0, 0, 0, 0]), qc)
        mujoco.mju_quat2Vel(er, qe, 1.0)
        w = d.xmat[pid].reshape(3, 3) @ d.qvel[3:6]
        tq = HARNESS_KR * er - HARNESS_KRD * w
        d.xfrc_applied[pid, :3] = f
        d.xfrc_applied[pid, 3:] = tq
        self.side.append(float(np.hypot(f[0], f[1])))
        self.torque.append(float(np.linalg.norm(tq)))
        return f

    def step(self):
        m, d = self.m, self.d
        k = min(int(d.time / CTRL_DT), len(self.times) - 2)
        u = (d.time - self.times[k]) / CTRL_DT
        q_des = (1 - u) * self.Q[k] + u * self.Q[k + 1]
        qd_des = (1 - u) * self.Qd[k] + u * self.Qd[k + 1]
        p_des = (1 - u) * self.P[k] + u * self.P[k + 1]
        share = (1 - u) * self.WS[k] + u * self.WS[k + 1]
        f = self.harness(p_des, self.Pd[k])
        tau = base.GAIN_SCALE * (base.KP * (q_des - d.qpos[self.leg_q]) + base.KD * (qd_des - d.qvel[self.leg_v]))
        for i, (b, w) in enumerate(zip(self.feet, share)):          # ขาที่ยืนดันพื้นรับน้ำหนักตัว (tau = J^T f)
            mujoco.mj_jacBody(m, d, self.jp, self.jr, b)
            cols = self.leg_v[6 * i:6 * i + 6]
            tau[6 * i:6 * i + 6] += self.jp[:, cols].T @ np.array([0.0, 0.0, -w * self.weight])
        d.ctrl[self.leg_act] = np.clip(tau, -base.TAU_LIMIT, base.TAU_LIMIT)
        mujoco.mj_step(m, d)
        self.fz.append(f[2])

    def report(self):
        top = self.heights.sum()
        z = self.d.xpos[self.pid][2]
        fz = np.clip(np.array(self.fz), 0, None) / self.weight * 100
        ok = z > top + 0.5
        print(f"ผล: {'ถึงชานพักด้านบน ✓' if ok else 'ไม่ถึงบนสุด ✗'}  สะโพกสูง {z:.2f} m")
        print(f"  สายแขวนช่วยยกตัว: เฉลี่ย {fz.mean():.0f}% ของน้ำหนักตัว {self.weight / 9.81:.0f} kg "
              f"(สูงสุด {fz.max():.0f}% ตอนวินาทีที่ {np.argmax(fz) * 0.001:.1f}) — ขารับน้ำหนักเองเป็นหลัก")
        print(f"  สายแขวนช่วยทรงตัว: แรงดันด้านข้าง/หน้าหลังเฉลี่ย {np.mean(self.side):.0f} N, "
              f"แรงบิดตั้งลำตัวเฉลี่ย {np.mean(self.torque):.0f} N·m — ส่วนนี้คือสิ่งที่ RL policy ต้องทำเองถ้าไม่มีสาย")
        return ok, fz.mean()


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="p3", choices=["p2", "p3"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--hmin", type=float, default=0.10)
    ap.add_argument("--hmax", type=float, default=0.20)
    ap.add_argument("--dmin", type=float, default=0.25)
    ap.add_argument("--dmax", type=float, default=0.40)
    ap.add_argument("--speed", type=float, default=1.0, help="ความเร็วการเล่นในหน้าต่าง 3D (0.5 = ช้าลงครึ่งหนึ่ง)")
    ap.add_argument("--record", help="บันทึกวิดีโอ .mp4 แทนการเปิดหน้าต่าง (ต้องมี ffmpeg)")
    ap.add_argument("--no-loop", action="store_true", help="ไม่วนกลับไปเริ่มใหม่เมื่อถึงบนสุด")
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else int(time.time()) % 10000
    heights, depths = stairs.make_stairs(np.random.default_rng(seed), a.steps, a.hmin, a.hmax, a.dmin, a.dmax)
    path = os.path.join(a.repo, "resources", "robots", "ATOM", "mjcf", f"atom_{a.model}_12dof.xml")
    m, d = build_scene(path, heights, depths)
    print(f"บันได {a.steps} ขั้น seed={seed}: สูง {heights.min() * 100:.1f}-{heights.max() * 100:.1f} ซม. "
          f"ลึก {depths[:-1].min() * 100:.1f}-{depths[:-1].max() * 100:.1f} ซม. รวมสูง {heights.sum() * 100:.0f} ซม.")
    c = Climber(m, d, heights, depths)
    pid = c.pid

    if a.record:
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        r = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        cam.distance, cam.azimuth, cam.elevation = 3.6, 110, -12
        fps = 30
        ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                               "-s", "1280x720", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "veryfast",
                               "-pix_fmt", "yuv420p", "-crf", "23", a.record], stdin=subprocess.PIPE)
        next_frame = 0.0
        while d.time < c.t_end:
            c.step()
            if d.time >= next_frame:
                cam.lookat[:] = [d.xpos[pid][0] + 0.3, 0, d.xpos[pid][2] - 0.35]
                r.update_scene(d, cam)
                ff.stdin.write(r.render().tobytes())
                next_frame += 1 / fps
        ff.stdin.close()
        ff.wait()
        c.report()
        print("บันทึกวิดีโอ", a.record)
        return

    from mujoco import viewer
    with viewer.launch_passive(m, d) as v:
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 3.6, 110, -12
        while v.is_running():
            t0 = time.time()
            for _ in range(10):
                c.step()
            v.cam.lookat[:] = [d.xpos[pid][0] + 0.3, 0, d.xpos[pid][2] - 0.35]
            v.sync()
            if d.time >= c.t_end:
                c.report()
                if a.no_loop:
                    while v.is_running():
                        time.sleep(0.05)
                    break
                time.sleep(1.5)
                c.reset()
            time.sleep(max(0.0, 0.01 / a.speed - (time.time() - t0)))


if __name__ == "__main__":
    main()
