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


# iterate through data
for dat_file in ecg_files:
    base = os.path.splitext(dat_file)[0]

    try:
        record = wfdb.rdrecord(base)
        signal = record.p_signal
        lead_names = record.sig_name

        plt.figure(figsize=(12, 6))

        # plot all leads at once
        for i in range(record.n_sig):
            plt.plot(signal[:, i], label=lead_names[i])

        plt.title(f"ECG Leads: {os.path.basename(base)}")
        plt.xlabel("Samples")
        plt.ylabel("Amplitude (mV)")
        plt.legend(loc='upper right')


        # Create a filename based on base name and lead name
        img_filename = f"{os.path.basename(base)}.png"
        img_path = os.path.join(output_dir, img_filename)
        plt.savefig(img_path)
        plt.close()
        print(f"Saved: {img_path}")
    except Exception as e:
        print(f"Failed to process {base}: {e}")
