import numpy as np

def TEST_signature_match(active_idx, signature_idx, method_name, pilot_type):
    misses = np.setdiff1d(signature_idx, active_idx)
    nr_of_misses = len(misses)
    percentage = (len(signature_idx) - nr_of_misses) / len(signature_idx) * 100
    print(f"Signature match for {method_name} {pilot_type}: {percentage:.3f}%")
    return percentage


