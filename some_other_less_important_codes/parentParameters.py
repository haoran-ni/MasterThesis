hypers_lr = {
    'smearing':0.7,
    'max_angular':0,
    'max_radial':10,
    'cutoff_radius':5.,
    'potential_exponent':6,
    'radial_basis': 'gto',
    'compute_gradients':True
}

hypers_rascaline = {
    "cutoff": 5.0,
    "atomic_gaussian_width": 0.7,
    "max_radial": 6,
    "max_angular": 6,
    "radial_basis": {"Gto": {}},
    "cutoff_function": {"ShiftedCosine": {"width": 1.}},
    "gradients": True
}

alpha = 1e-1