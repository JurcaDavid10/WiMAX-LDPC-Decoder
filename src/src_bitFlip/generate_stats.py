import numpy as np
import csv
from bitFlip import load_and_expand_matrix, gdbf_decode, file_content

def run_simulation():
    print("1. Expanding WiMAX Matrix (Z=96)...")
    H = load_and_expand_matrix(file_content, Z=96)
    
    # Test range: 4.0dB to 6.0dB
    # Increased range slightly to capture the 'waterfall'
    snr_range = np.linspace(4.0, 6.0, 6)
    num_trials = 50
    results = []

    print("2. Running BER simulation (Endurance mode: 500 iterations)...")
    for eb_n0 in snr_range:
        successes = 0
        rate = 5/6
        snr_linear = 10 ** (eb_n0 / 10)
        sigma = np.sqrt(1 / (2 * rate * snr_linear))
        
        for i in range(num_trials):
            # Generate new noise for every trial
            noise = sigma * np.random.randn(H.shape[1])
            received_signal = np.ones(H.shape[1]) + noise
            
            # --- INCREASED ENDURANCE: max_iter=500 ---
            _, success = gdbf_decode(H, received_signal, max_iter=500, probabilistic=True)
            
            if success: 
                successes += 1
        
        ber = (num_trials - successes) / num_trials
        results.append([eb_n0, ber])
        print(f"SNR: {eb_n0:.1f} dB | Success Rate: {successes/num_trials*100}% | BER: {ber}")

    with open('simulation_results.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['SNR_dB', 'BER'])
        writer.writerows(results)
    print("\nDone! Results saved to simulation_results.csv")

if __name__ == "__main__":
    run_simulation()