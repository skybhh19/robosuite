"""Opt-in grasp-only contact compliance; no friction/geometry/mass changes."""
import xml.etree.ElementTree as ET
import numpy as np

CONTACT_DYNAMICS = {'legacy': None, 'damped': (.015, 1.), 'firm': (.006, 1.),
                    'low_slip': (.015, .75), 'low_slip_stiff': (.015, .75),
                    'low_slip_damped': (.015, 1.), 'low_slip_firm': (.006, 1.),
                    'low_slip_stiff_damped': (.015, 1.), 'low_slip_stiff_firm': (.006, 1.)}


def add_grasp_contact_pairs(contact, mode, finger_torsion):
    if mode not in CONTACT_DYNAMICS:
        raise ValueError('Unknown fork pad contact dynamics: ' + str(mode))
    if mode == 'legacy':
        return
    for surface in ('handle', 'neck'):
        for finger in (1, 2):
            name = f'fork_grasp_{surface}_finger{finger}'
            if contact.find(f"pair[@name='{name}']") is not None:
                raise ValueError('Duplicate grasp contact pair')
            pair = ET.SubElement(contact, 'pair', name=name,
                geom1=f'fork_{surface}_collision',
                geom2=f'gripper0_right_finger{finger}_pad_collision',
                condim='4', friction=f'1.5 1.5 {finger_torsion} 0.0001 0.0001',
                solref=' '.join(map(str, CONTACT_DYNAMICS[mode])),
                solimp='0.9 0.95 0.001 0.5 2')
            if mode.startswith('low_slip'):
                # Stiffen the friction constraints rather than raising their
                # force limit. MuJoCo's soft friction otherwise permits creep.
                pair.set('solreffriction', '0.004 1')
                pair.set('solimp', '0.99 0.999 0.001 0.5 2')
                if mode in ('low_slip_stiff', 'low_slip_stiff_damped', 'low_slip_stiff_firm'):
                    pair.set('solimp', '0.999 0.9999 0.001 0.5 2')


def contact_diagnostics(env):
    import mujoco
    normal = torsion = 0.
    table = False
    contacts = 0
    minimum_distance = 0.
    for index, contact in enumerate(env.sim.data.contact):
        names = [env.sim.model.geom_names[g] or '' for g in (contact.geom1, contact.geom2)]
        if not any(n.startswith('fork_') for n in names):
            continue
        table |= any('table' in n for n in names)
        if not any('finger' in n or 'pad' in n for n in names):
            continue
        force = np.zeros(6)
        mujoco.mj_contactForce(env.sim.model._model, env.sim.data._data, index, force)
        normal += max(0., float(force[0]))
        torsion += abs(float(force[3]))
        minimum_distance = min(minimum_distance, float(contact.dist))
        contacts += 1
    return dict(fork_finger_normal_force_N=normal, fork_finger_torsion_abs_Nm=torsion,
                fork_finger_contact_count=contacts, fork_table_contact=table,
                fork_finger_minimum_distance_m=minimum_distance)
