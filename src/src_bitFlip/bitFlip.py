import numpy as np
import math
import io

# ==========================================
# THE WIMAX BASE MATRIX FILE
# ==========================================

file_content = """nb_rows     4
nb_cols    24
max_col_deg   4
max_row_deg  20

qc_base_matrix
 1 25 55 -1 47  4 -1 91 84  8 86 52 82 33  5  0 36 20  4 77 80  0 -1 -1
-1  6 -1 36 40 47 12 79 47 -1 41 21 12 71 14 72  0 44 49  0  0  0  0 -1
51 81 83  4 67 -1 21 -1 31 24 91 61 81  9 86 78 60 88 67 15 -1 -1  0  0
50 -1 50 15 -1 36 13 10 11 20 53 90 29 92 57 30 84 92 11 66 80 -1 -1  0
* -------------------------------------------------------------------------
* Each non-negative entry B(i,j) >= 0 is replaced by a circular permutation 
* matrix of size Z (called 'expansion factor'),  obtained by right-shifting 
* the identity matrix.
* -------------------------------------------------------------------------"""

# ==========================================
# PARSE AND EXPAND MATRIX
# ==========================================
def load_and_expand_matrix(file_text, Z=96):
    lines = file_text.strip().split('\n')
    base_matrix = []
    
    # Extract only the matrix rows (lines starting with numbers or -1)
    for line in lines:
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Filter out the header variables
            if "nb_rows" not in line and "nb_cols" not in line:
                row_vals = [int(val) for val in line.split()]
                if len(row_vals) == 24: # Ensure it's a matrix row
                    base_matrix.append(row_vals)
                    
    base_matrix = np.array(base_matrix)
    rows_b, cols_b = base_matrix.shape
    
    # Initialize the expanded full matrix H (384 x 2304)
    H = np.zeros((rows_b * Z, cols_b * Z), dtype=int)
    
    # Expand based on the shift rule
    for i in range(rows_b):
        for j in range(cols_b):
            val = base_matrix[i, j]
            if val >= 0:
                shift = math.floor(val * Z / 96) # For Z=96, this is just 'val'
                I = np.eye(Z, dtype=int)
                # circularly right-shift the identity matrix
                I_shifted = np.roll(I, shift, axis=1) 
                H[i*Z:(i+1)*Z, j*Z:(j+1)*Z] = I_shifted
                
    return H

# ==========================================
# GDBF DECODER ALGORITHM
# ==========================================
def gdbf_decode(H, y, max_iter=100, probabilistic=False):
    """
    H: Parity Check Matrix (384 x 2304)
    y: Received soft channel values
    """
    # Step 1: Initial hard decision (Bipolar: +1 or -1)
    x = np.sign(y)
    x[x == 0] = 1 # Handle edge case of exact 0.0
    
    for iteration in range(max_iter):
        
        # Step 2: Parity Check
        # Python Trick for speed: instead of looping 384 times to multiply bipolar bits,
        # we convert temporarily to binary (0/1) to use fast matrix multiplication modulo 2.
        # Bipolar +1 -> Binary 0. Bipolar -1 -> Binary 1.
        x_bin = (1 - x) // 2
        syndrome = (H @ x_bin) % 2
        
        # If syndrome is all zeros, ALL checks passed!
        if np.sum(syndrome) == 0:
            print(f"-> SUCCESS! Decoder converged at iteration {iteration + 1}")
            return x_bin, True 
            
        # Convert binary syndrome back to bipolar check node statuses
        # Binary 0 (Pass) -> +1. Binary 1 (Fail) -> -1.
        C = 1 - 2 * syndrome 
        
        # Step 3: Compute Energy Vector for all 2304 bits
        # H.T @ C gives the sum of connected checks for every bit instantly
        chk_sum = H.T @ C
        E = (x * y) + chk_sum

        # Step 4: Gradient Descent Flip
        # Add a 0.5 tolerance so we group all bits with the worst parity checks
        theta = np.min(E)                     # Find lowest energy
        target_bits = (E <= theta + 0.5)      # Find indices of bits with lowest energy
        
        if probabilistic:
            # PGDBF: 80% chance to flip bits with minimum energy
            # Breaks out of trapping sets!
            for idx in np.where(target_bits)[0]:
                if np.random.rand() < 0.80:
                    x[idx] *= -1
        else:
            # Standard GDBF: Flip all bits with minimum energy unconditionally
            x[target_bits] *= -1 

    # Calculate how many errors are left before giving up
    remaining = np.sum(((1 - x) // 2) != 0)
    print(f"-> FAILURE: Reached max iterations. Bits still wrong: {remaining}")
    return (1 - x) // 2, False

# ==========================================
# MAIN SIMULATION PIPELINE
# ==========================================
if __name__ == "__main__":
    np.random.seed(42) # For reproducible results
    
    print("1. Expanding WiMAX Matrix (Z=96)...")
    H = load_and_expand_matrix(file_content, Z=96)
    print(f"   Expanded Parity Matrix Shape: {H.shape}\n")
    
    # 2. Simulate Transmission
    # For linear block codes, we can test decoding using the all-zero codeword.
    # BPSK mapping: Binary 0 -> Transmit +1
    N = H.shape[1]
    transmitted_signal = np.ones(N) 
    
    # 3. Add Channel Noise (AWGN)
    # Let's use a moderate Signal-to-Noise Ratio (Eb/N0 = 4.5 dB)
    Eb_N0_dB = 4.5
    rate = 5/6
    snr_linear = 10 ** (Eb_N0_dB / 10)
    sigma = np.sqrt(1 / (2 * rate * snr_linear))
    
    noise = sigma * np.random.randn(N)
    received_signal = transmitted_signal + noise
    
    # Count how many errors the channel introduced before decoding
    initial_errors = np.sum(np.sign(received_signal) < 0)
    print(f"2. Channel Simulation (Eb/N0 = {Eb_N0_dB} dB)")
    print(f"   Initial Bit Errors from Channel: {initial_errors} out of 2304")
    
    # === NEW: Find and print the exact locations of the channel errors ===
    error_indices = np.where(received_signal < 0)[0]
    print(f"   First 10 corrupted bit locations: {error_indices[:10]}\n")
    
    # 4. Run the Decoder
    print("3. Starting GDBF Decoding...")
    
    # NOTE: Set probabilistic=True to use PGDBF (highly recommended for WiMAX)
    decoded_bits, success = gdbf_decode(H, received_signal, max_iter=100, probabilistic=True)
    
    # 5. Final validation
    if success:
        # Check against our original all-zero codeword
        remaining_errors = np.sum(decoded_bits != 0)
        print(f"   Final Remaining Errors: {remaining_errors}")
        
        # === NEW: Show what the decoder output at those exact corrupted locations ===
        fixed_bits = decoded_bits[error_indices[:10]]
        print(f"   Decoded values at those locations: {fixed_bits}")