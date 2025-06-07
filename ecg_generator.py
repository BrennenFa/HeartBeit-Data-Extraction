import os
import glob
import wfdb
import numpy as np
import matplotlib.pyplot as plt
 

# find all data
ecg_files = glob.glob('data/**/*.dat', recursive=True)
output_dir = 'ecg_images'
os.makedirs(output_dir, exist_ok=True)
print("success")



# iterate through datat
for dat_file in ecg_files:
    # base id
    base = os.path.splitext(dat_file)[0]

    try:
        record = wfdb.rdrecord(base)

        for i in range(record.n_sig):
            lead_name = record.sig_name[i]
            signal = record.p_signal[:, i]

            # Plot this individual lead
            plt.figure(figsize=(12, 3))
            plt.plot(signal)
            plt.title(f"ECG Lead: {lead_name} ({os.path.basename(base)})")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude (mV)")

            # Create a filename based on base name and lead name
            img_filename = f"{os.path.basename(base)}_{lead_name}.png"
            img_path = os.path.join(output_dir, img_filename)
            plt.savefig(img_path)
            plt.close()
            print(f"Saved: {img_path}")
    except Exception as e:
        print(f"Failed to process {base}: {e}")
