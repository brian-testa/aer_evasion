# Run at command line with sudo
import subprocess

for sample_size in [50,30]:
    sample_str = f'{sample_size}_{sample_size}_{200-2*sample_size}'

    for ae_str in ["PURE", "MSE", "MSE-FFT"]:
        for trial_num in range(10):
            conf = "{\"job0\": {\"runtime_str\": \"TESS_UNTARGETED_{0}_AE_{1}_TRIAL_{2}\",".format(sample_str, ae_str, f'{trial_num:02}')
            conf += "\"method\": \"kfold_sample_run\","
            conf += "\"params\": { "
            conf += "\"modality\": \"RECORDED}\", "
            conf += "\"sample_size\": {0}, ".format(sample_size*2)
            conf += "\"num_trials\": 1, "
            conf += "\"dataset_str\": \"TESS\", "
            conf += "\"autoencoder_key\": \"{0}\", ".format(ae_str)
            conf += "\"target_class\": -1, "
            conf += "\"starting_population\": \"/pickles/populations_RAVDESS_UNTARGETED_CONTINUED.pickle", "
            conf += "\"starting_generation\": -1, "
            conf += "\"ngen\": 10 "
            conf += "} } }\n"

            f = open(f"/tmp/dig_run_ss_{sample_str}_ae_{ae_str}_trial_{trial_num:02}_config.json", "w")
            f.write(conf)
            f.close()

            print(conf)
            subprocess.Popen(["sudo", "docker", "run", "-v", "/home/brian/Workspace/AUDIO_FINAL/pickles:/pickles", "-v", "/home/brian/Workspace/AUDIO_FINAL/data:/data", "audio", "python3", "scripts/batch_run.py", f"/tmp/dig_run_ss_{sample_str}_ae_{ae_str}_trial_{trial_num:02}_config.json"])
