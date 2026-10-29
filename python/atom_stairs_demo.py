"""
atom_stairs_demo.py — บันได 10 ขั้นแบบสุ่ม (สูง/ลึกไม่เท่ากัน) หน้าหุ่น DOBOT Atom ใน MuJoCo

ต้องวางไฟล์นี้ไว้โฟลเดอร์เดียวกับ atom_stand_demo.py (ใช้ตัวควบคุมการยืนตัวเดียวกัน)
หุ่นยังขึ้นบันไดไม่ได้ — ต้องใช้ RL policy ที่เทรนแล้ว ฉากนี้ไว้เป็นสนามทดสอบ (Sim2Sim)

วิธีใช้:
    python atom_stairs_demo.py --repo .\\atom-locomotion-training-main               (เปิดหน้าต่าง 3D)
    python atom_stairs_demo.py --repo ... --seed 7                                    (สุ่มบันไดชุดใหม่)
    python atom_stairs_demo.py --repo ... --hmin 0.08 --hmax 0.15 --dmin 0.28 --dmax 0.35
    python atom_stairs_demo.py --repo ... --snapshot stairs.png                       (บันทึกภาพแทน)
    python atom_stairs_demo.py --repo ... --save-xml stairs_scene.xml                 (บันทึกฉากเป็นไฟล์ MJCF)
"""
import argparse
import os
import sys
import time

import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import atom_stand_demo as base  # noqa: E402

STAIR_WIDTH = 1.2      # m ความกว้างบันได (แกน y)
START_X = 0.55         # m ระยะจากจุดยืนหุ่นถึงขั้นแรก
LANDING = 1.0          # m ชานพักด้านบน


def make_stairs(rng, n, hmin, hmax, dmin, dmax):
    h = rng.uniform(hmin, hmax, n)
    d = rng.uniform(dmin, dmax, n)
    d[-1] = LANDING
    return h, d


def add_stairs(spec, heights, depths):
    """ใส่บันไดเป็นก้อนกล่องตัน: ขั้นที่ i สูงจากพื้นถึงความสูงสะสม H_i"""
    x, top = START_X, 0.0
    colors = ([0.78, 0.62, 0.45, 1], [0.70, 0.55, 0.40, 1])
    for i, (h, d) in enumerate(zip(heights, depths)):
        top += h
        g = spec.worldbody.add_geom()
        g.name = f"stair_{i + 1}"
        g.type = mujoco.mjtGeom.mjGEOM_BOX
        g.size = [d / 2, STAIR_WIDTH / 2, top / 2]
        g.pos = [x + d / 2, 0, top / 2]
        g.rgba = colors[i % 2]
        g.friction = [1.0, 0.1, 0.1]
        g.contype = g.conaffinity = 1
        x += d
    return x


def build(model_path, heights, depths):
    spec = mujoco.MjSpec.from_file(model_path)
    add_stairs(spec, heights, depths)
    m = spec.compile()
    m.opt.timestep = 0.001
    d = mujoco.MjData(m)
    leg_q = [m.jnt_qposadr[m.joint(n).id] for n in base.LEG_JOINTS]
    leg_v = [m.jnt_dofadr[m.joint(n).id] for n in base.LEG_JOINTS]
    leg_act = [m.actuator(n).id for n in base.LEG_JOINTS]
    upper = [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name not in base.LEG_JOINTS]
    up_q = [m.jnt_qposadr[j] for j in upper]
    up_v = [m.jnt_dofadr[j] for j in upper]
    d.qpos[leg_q] = base.DEFAULT
    mujoco.mj_forward(m, d)
    return spec, m, d, (leg_q, leg_v, leg_act, up_q, up_v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".")
    ap.add_argument("--model", default="p3", choices=["p2", "p3"])
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=None, help="เลขสุ่ม ใส่เลขเดิมจะได้บันไดชุดเดิม")
    ap.add_argument("--hmin", type=float, default=0.10, help="ความสูงขั้นต่ำสุด (m)")
    ap.add_argument("--hmax", type=float, default=0.20, help="ความสูงขั้นสูงสุด (m)")
    ap.add_argument("--dmin", type=float, default=0.25, help="ความลึกขั้นต่ำสุด (m)")
    ap.add_argument("--dmax", type=float, default=0.40, help="ความลึกขั้นสูงสุด (m)")
    ap.add_argument("--snapshot")
    ap.add_argument("--save-xml")
    a = ap.parse_args()

    seed = a.seed if a.seed is not None else int(time.time()) % 10000
    rng = np.random.default_rng(seed)
    heights, depths = make_stairs(rng, a.steps, a.hmin, a.hmax, a.dmin, a.dmax)

    path = os.path.join(a.repo, "resources", "robots", "ATOM", "mjcf", f"atom_{a.model}_12dof.xml")
    spec, m, d, idx = build(path, heights, depths)

    print(f"บันได {a.steps} ขั้น  seed={seed}  (ใช้ --seed {seed} เพื่อได้ชุดนี้อีกครั้ง)")
    print(" ขั้น  สูง(ซม.)  ลึก(ซม.)  สูงสะสม(ซม.)")
    for i, (h, dd) in enumerate(zip(heights, depths)):
        tag = "  (ชานพัก)" if i == len(heights) - 1 else ""
        print(f"  {i + 1:>2}    {h * 100:5.1f}    {dd * 100:5.1f}      {heights[:i + 1].sum() * 100:6.1f}{tag}")
    foot = 24.0
    short = [i + 1 for i, dd in enumerate(depths[:-1]) if dd * 100 < foot]
    if short:
        print(f"  หมายเหตุ: ขั้นที่ {short} ลึกน้อยกว่าความยาวเท้า Atom (~{foot:.0f} ซม.) เท้าจะยื่นพ้นขอบ")

    if a.save_xml:
        with open(a.save_xml, "w", encoding="utf-8") as f:
            f.write(spec.to_xml())
        print("บันทึกฉาก", a.save_xml, "(เก็บไว้ในโฟลเดอร์ mjcf เพื่อให้หา mesh เจอ)")

    if a.snapshot:
        while d.time < 3.0:
            base.control(m, d, idx, squat=False)
            mujoco.mj_step(m, d)
        m.vis.global_.offwidth, m.vis.global_.offheight = 1280, 720
        r = mujoco.Renderer(m, 720, 1280)
        cam = mujoco.MjvCamera()
        total = START_X + depths.sum()
        cam.lookat[:] = [total / 2 - 0.3, 0, heights.sum() / 2 + 0.3]
        cam.distance, cam.azimuth, cam.elevation = total * 0.85 + 1.0, 105, -12
        r.update_scene(d, cam)
        import PIL.Image
        PIL.Image.fromarray(r.render()).save(a.snapshot)
        print("บันทึกภาพ", a.snapshot)
        return

    from mujoco import viewer
    with viewer.launch_passive(m, d) as v:
        v.cam.lookat[:] = [1.6, 0, 0.8]
        v.cam.distance, v.cam.azimuth, v.cam.elevation = 4.5, 105, -12
        while v.is_running():
            t0 = time.time()
            for _ in range(10):
                base.control(m, d, idx, squat=False)
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0, 0.01 - (time.time() - t0)))


if __name__ == "__main__":
    main()
