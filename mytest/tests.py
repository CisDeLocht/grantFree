import numpy as np

def TEST_signature_match(active_idx, signature_idx, method_name, pilot_type):
    misses = np.setdiff1d(signature_idx, active_idx)
    nr_of_misses = len(misses)
    percentage = (len(signature_idx) - nr_of_misses) / len(signature_idx) * 100
    print(f"Signature match for {method_name} {pilot_type}: {percentage:.3f}%")
    return percentage

def TEST_normed(pilots):
    check = np.ones(pilots.shape[1])
    norms = np.linalg.norm(pilots, axis=0)
    tolerance = 1e-6
    boolie = all(abs(el1 - el2) < tolerance for el1, el2 in zip(check, norms))
    return boolie