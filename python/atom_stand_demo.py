"""
atom_stand_demo.py — ลองเล่นหุ่น DOBOT Atom ใน MuJoCo โดยไม่ต้องใช้ Isaac Gym / GPU

ใช้ไฟล์โมเดล MJCF จาก repo: github.com/embodied-dobot/atom-locomotion-training
หุ่นจะยืนด้วย PD controller ที่ขา 12 ข้อ (ค่า kp/kd เดียวกับ sim2sim.py ของ DOBOT)
แล้วย่อเข่าลง-ยืดขึ้นเป็นจังหวะ (squat)  — ยังไม่ใช่การเดิน การเดินต้องใช้ RL policy ที่เทรนแล้ว

วิธีใช้:
    pip install mujoco numpy
    python atom_stand_demo.py --repo C:\\path\\to\\atom-locomotion-training          (เปิดหน้าต่าง 3D)
    python atom_stand_demo.py --repo ... --model p2                                  (เลือกรุ่น p2/p3)
    python atom_stand_demo.py --repo ... --no-squat                                  (ยืนนิ่ง)
    python atom_stand_demo.py --repo ... --snapshot atom.png                         (ไม่เปิดหน้าต่าง บันทึกภาพ)

ในหน้าต่าง viewer: ลากเมาส์ซ้าย = หมุนกล้อง, ขวา = เลื่อน, scroll = ซูม,
ดับเบิลคลิกชิ้นส่วน แล้ว Ctrl+ลากขวา = ผลักหุ่น, Space = pause, Backspace = reset
"""
import argparse
import math
import os
import time

import mujoco
import numpy as np

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
# ท่ายืนตั้งต้น (rad) และ PD gains — คัดลอกจาก atom_p3.py / sim2sim.py ของ DOBOT
DEFAULT = np.array([-0.2, 0, 0, 0.4, -0.2, 0, -0.2, 0, 0, 0.4, -0.2, 0])
KP = np.array([100, 95, 90, 110, 50, 50, 100, 95, 90, 110, 50, 50], dtype=float)
KD = np.array([3.5, 2.5, 2.5, 3.5, 1.5, 1.5, 3.5, 2.5, 2.5, 3.5, 1.5, 1.5])
TAU_LIMIT = 200.0
# ค่า KP/KD ของ DOBOT ออกแบบมาคู่กับ RL policy ที่คอยทรงตัวให้ ถ้าใช้ PD เปล่า ๆ หุ่นจะล้มใน ~2 วินาที
# เดโมนี้จึงคูณ gain ให้แข็งขึ้น และเพิ่มการทรงตัวแบบง่ายที่ข้อเท้า (ankle strategy)
GAIN_SCALE = 4.0
BAL_P, BAL_D = 1.0, 0.1   # rad ของข้อเท้า ต่อ rad ที่ลำตัวเอียง / ต่อ rad/s
# ในโมเดล 12dof ของ DOBOT แขน/หัว/เอวถูกล็อกไว้ (ขยับได้เฉพาะขา) — ถ้าคุณปลดล็อกข้อต่อเหล่านั้นเอง
# ค่า PD เสมือนนี้จะกันไม่ให้แขนห้อยแกว่ง
UPPER_KP, UPPER_KD = 40.0, 2.0


def build(model_path):
    m = mujoco.MjModel.from_xml_path(model_path)
    m.opt.timestep = 0.001
    d = mujoco.MjData(m)
    leg_q = [m.jnt_qposadr[m.joint(n).id] for n in LEG_JOINTS]
    leg_v = [m.jnt_dofadr[m.joint(n).id] for n in LEG_JOINTS]
    leg_act = [m.actuator(n).id for n in LEG_JOINTS]
    upper = [j for j in range(m.njnt)
             if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE and m.joint(j).name not in LEG_JOINTS]
    up_q = [m.jnt_qposadr[j] for j in upper]
    up_v = [m.jnt_dofadr[j] for j in upper]
    d.qpos[leg_q] = DEFAULT
    mujoco.mj_forward(m, d)
    return m, d, (leg_q, leg_v, leg_act, up_q, up_v)


def control(m, d, idx, squat):
    leg_q, leg_v, leg_act, up_q, up_v = idx
    target = DEFAULT.copy()
    if squat:  # ย่อเข่า 0.4 -> 1.0 rad เป็นจังหวะ 4 วินาที (สะโพก/ข้อเท้าชดเชยให้ลำตัวตั้งตรง)
        s = 0.5 * (1 - math.cos(2 * math.pi * d.time / 4.0)) * min(d.time / 2.0, 1.0)
        bend = 0.6 * s
        for o in (0, 6):
            target[o + 0] -= bend / 2
            target[o + 3] += bend
            target[o + 4] -= bend / 2
    w, x, y, z = d.qpos[3:7]                       # มุมก้ม-เงยของลำตัว (pitch) จาก quaternion
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    for o in (0, 6):
        target[o + 4] += BAL_P * pitch + BAL_D * d.qvel[4]
    tau = GAIN_SCALE * (KP * (target - d.qpos[leg_q]) - KD * d.qvel[leg_v])
    d.ctrl[leg_act] = np.clip(tau, -TAU_LIMIT, TAU_LIMIT)
    d.qfrc_applied[:] = 0
    d.qfrc_applied[up_v] = -UPPER_KP * d.qpos[up_q] - UPPER_KD * d.qvel[up_v]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=".", help="โฟลเดอร์ atom-locomotion-training")
    ap.add_argument("--model", default="p3", choices=["p2", "p3"])
    ap.add_argument("--no-squat", action="store_true")
    ap.add_argument("--snapshot", help="บันทึกภาพ PNG แทนการเปิดหน้าต่าง")
    ap.add_argument("--seconds", type=float, default=6.0, help="ใช้กับ --snapshot")
    a = ap.parse_args()

    path = os.path.join(a.repo, "resources", "robots", "ATOM", "mjcf", f"atom_{a.model}_12dof.xml")
    m, d, idx = build(path)
    print(f"โหลด {path}\n  bodies={m.nbody} joints={m.njnt} motors={m.nu} มวลรวม={sum(m.body_mass):.1f} kg")
    squat = not a.no_squat

    if a.snapshot:
        while d.time < a.seconds:
            control(m, d, idx, squat)
            mujoco.mj_step(m, d)
        print(f"  หลัง {a.seconds:.0f} วินาที ความสูงสะโพก = {d.qpos[2]:.3f} m (ล้มถ้า < 0.5)")
        m.vis.global_.offwidth, m.vis.global_.offheight = 960, 720
        r = mujoco.Renderer(m, 720, 960)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [d.qpos[0], d.qpos[1], 0.75]
        cam.distance, cam.azimuth, cam.elevation = 3.0, 135, -12
        r.update_scene(d, cam)
        import PIL.Image
        PIL.Image.fromarray(r.render()).save(a.snapshot)
        print("  บันทึกภาพ", a.snapshot)
        return

    from mujoco import viewer
    with viewer.launch_passive(m, d) as v:
        while v.is_running():
            t0 = time.time()
            for _ in range(10):  # 10 ms ต่อเฟรม
                control(m, d, idx, squat)
                mujoco.mj_step(m, d)
            v.sync()
            time.sleep(max(0, 0.01 - (time.time() - t0)))


if __name__ == "__main__":
    main()
