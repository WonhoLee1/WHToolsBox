import numpy as np
import pickle
import os
from typing import Optional

CONTACT_THRESHOLD = 1.0  # N
PEAK_WINDOW = 50  # frames

def extract_contact_sequence(result) -> dict:
    raw = getattr(result, 'corner_impact_hist', None)
    if raw is None:
        raise ValueError(
            "corner_impact_hist not found in result. "
            "Re-run simulation after Phase 0 schema update."
        )
    cih = np.asarray(raw)  # shape (N, 8)
    time_history = np.asarray(result.time_history)
    dt = float(time_history[1] - time_history[0]) if len(time_history) > 1 else 1e-3

    contacts = []
    for i in range(min(8, cih.shape[1])):
        force_series = cih[:, i]
        contact_indices = np.where(force_series > CONTACT_THRESHOLD)[0]
        if len(contact_indices) == 0:
            continue
        first_idx = contact_indices[0]
        pre_idx = max(0, first_idx - 1)
        contact_time = float(time_history[pre_idx])

        peak_end = min(first_idx + PEAK_WINDOW, len(force_series))
        peak_force = float(np.max(force_series[first_idx:peak_end]))

        # find end of contact
        post = force_series[first_idx:]
        below = np.where(post < CONTACT_THRESHOLD)[0]
        contact_end = first_idx + int(below[0]) if len(below) > 0 else len(force_series)
        impulse = float(np.trapz(force_series[first_idx:contact_end], dx=dt))

        contacts.append({
            'corner': f'C{i+1}',
            'time': contact_time,
            'force': peak_force,
            'impulse': impulse,
        })

    # sort by contact time
    contacts.sort(key=lambda x: x['time'])

    seq_str = '-'.join(c['corner'] for c in contacts)
    time_list   = [[c['corner'], c['time']]    for c in contacts]
    force_list  = [[c['corner'], c['force']]   for c in contacts]
    impact_list = [[c['corner'], c['impulse']] for c in contacts]

    result_dict = {
        'ContactSeqStr':      seq_str,
        'ContactSeqTimeList':  time_list,
        'ContactSeqForceList': force_list,
        'ContactSeqImpactList': impact_list,
    }

    # attach to result and re-save pkl if possible
    result.contact_seq = result_dict
    _try_save_result(result)

    # try to capture pre-contact frames
    _try_capture_contacts(result, contacts)

    return result_dict


def _try_save_result(result):
    pkl_path = getattr(result, 'pkl_path', None)
    if pkl_path and os.path.exists(pkl_path):
        try:
            with open(pkl_path, 'wb') as f:
                pickle.dump(result, f)
        except Exception as e:
            print(f"[whts_contact_extractor] Warning: could not re-save pkl: {e}")


def _try_capture_contacts(result, contacts):
    output_dir = getattr(result, 'output_dir', None)
    if output_dir is None:
        return
    contacts_dir = os.path.join(str(output_dir), 'contacts')
    os.makedirs(contacts_dir, exist_ok=True)
    try:
        import mujoco
        model = getattr(result, 'model', None)
        data = getattr(result, 'data', None)
        if model is None or data is None:
            return
        with mujoco.Renderer(model) as renderer:
            for c in contacts:
                frame_idx = _find_pre_contact_frame(result, c['corner'])
                if frame_idx is None:
                    continue
                renderer.update_scene(data)
                img = renderer.render()
                import imageio
                out_path = os.path.join(contacts_dir, f"{c['corner']}_pre.png")
                imageio.imwrite(out_path, img)
                print(f"[whts_contact_extractor] Saved {out_path}")
    except Exception as e:
        print(f"[whts_contact_extractor] Warning: capture failed (headless/no model): {e}")


def _find_pre_contact_frame(result, corner_name):
    i = int(corner_name[1:]) - 1  # C1→0, C4→3
    raw = getattr(result, 'corner_impact_hist', None)
    if raw is None:
        return None
    cih = np.asarray(raw)
    if i >= cih.shape[1]:
        return None
    force_series = cih[:, i]
    contact_indices = np.where(force_series > CONTACT_THRESHOLD)[0]
    if len(contact_indices) == 0:
        return None
    return max(0, contact_indices[0] - 1)
