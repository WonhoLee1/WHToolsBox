import os
import sys
import time
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap

# UTF-8 설정 (이모지 지원)
if sys.stdout.encoding != 'utf-8':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    except:
        pass

def _compute_batch_metrics_jax_standalone(q_hist, root_id, body_ids, nom_mats):
    """JAX 버전 구조 해석 엔진 유닛 테스트"""
    def quat_to_mat(q):
        w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
        return jnp.stack([
            jnp.stack([1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w], axis=-1),
            jnp.stack([2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w], axis=-1),
            jnp.stack([2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2], axis=-1)
        ], axis=-2)

    mats = quat_to_mat(q_hist)
    inv_root_mats = jnp.transpose(mats[:, root_id], (0, 2, 1))
    target_mats = mats[:, body_ids]
    
    # [Fix V5.2.8.5] Broadcasting matmul
    rel_mats = jnp.matmul(inv_root_mats[:, jnp.newaxis, :, :], target_mats)
    dev_mats = jnp.matmul(jnp.transpose(nom_mats, (0, 2, 1))[jnp.newaxis, :, :, :], rel_mats)
    
    bend = jnp.degrees(jnp.arccos(jnp.clip(dev_mats[:, :, 2, 2], -1.0, 1.0)))
    return bend

def _compute_batch_metrics_standard_standalone(q_hist, root_id, body_ids):
    """NumPy 버전 구조 해석 (Loop 기반)"""
    import scipy.spatial.transform as sst
    F = q_hist.shape[0]
    N = len(body_ids)
    results = np.zeros((F, N))
    
    for f in range(F):
        # Rotation logic (sst quat format is [x,y,z,w])
        root_q = q_hist[f, root_id]
        root_rot_inv = sst.Rotation.from_quat([root_q[1], root_q[2], root_q[3], root_q[0]]).inv()
        for i, b_id in enumerate(body_ids):
            target_q = q_hist[f, b_id]
            target_rot = sst.Rotation.from_quat([target_q[1], target_q[2], target_q[3], target_q[0]])
            rel_rot = root_rot_inv * target_rot
            # Simplified check for timing parity
            results[f, i] = 0.0 
    return results

# 1. 테스트 데이터 생성 (1000 프레임, 64개 블록)
F, N = 1000, 64
print(f"\n[WHTOOLS] JAX vs NumPy Benchmark Initialization...")
print(f" - Frames: {F}")
print(f" - Blocks: {N}")

q_data = np.random.randn(F, 100, 4)
q_data /= np.linalg.norm(q_data, axis=-1, keepdims=True)
root_id = 0
body_ids = np.arange(1, 65)
nom_mats = jnp.array([jnp.eye(3) for _ in range(64)])

# 2. NumPy 벤치마크
t0 = time.perf_counter()
_compute_batch_metrics_standard_standalone(q_data, root_id, body_ids)
dt_np = time.perf_counter() - t0
print(f" > Standard Engine (NumPy): {dt_np:8.4f} sec")

# 3. JAX 벤치마크 (Warmup + Run)
q_jax = jnp.array(q_data)
_compute_batch_metrics_jax_standalone(q_jax, root_id, body_ids, nom_mats).block_until_ready()

t1 = time.perf_counter()
_compute_batch_metrics_jax_standalone(q_jax, root_id, body_ids, nom_mats).block_until_ready()
dt_jax = time.perf_counter() - t1
print(f" > Accelerated Engine (JAX): {dt_jax:8.4f} sec")

# 4. 결과 출력
print("-" * 60)
speedup = dt_np / dt_jax if dt_jax > 1e-9 else 0
print(f" 🚀 JAX Speedup: {speedup:6.2f}x Faster")
print("=" * 60 + "\n")
