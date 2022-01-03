# Run at command line with sudo
import subprocess

for pert_indx in range(20):
    for loudness in [17, 16, 14, 12, 9]:
        conf = "{\"job0\": {\"input_pickle\": \"/pickles/recorded_perturbations.pickle\","
        conf += "\"dataset_str\": \"TESS\","
        conf += "\"mix_eval_filename\":\"/home/brian/Workspace/AUDIO_FINAL/pickles/dig_mix_pi_{0}_l_{1}.pickle\",".format(pert_indx, loudness)
        conf += "\"params\": { \"sample_sz_strs\": [], \"autoencoder_keys\": [], \"trial_numbers\": [],"
        conf += "\"population_indices\": [], \"perturbation_indices\": [{0}],".format(pert_indx)
        conf += "\"loudnesses\": [{0}]".format(loudness)
        conf += "} } }\n"

        f = open(f"/data/dig_mix_pi_{pert_indx}_l_{loudness}_config.json", "w")
        f.write(conf)
        f.close()

        print(conf)
        subprocess.Popen(["sudo", "docker", "run", "--rm", "-v", "/home/brian/Workspace/AUDIO_FINAL/pickles:/pickles", "-v", "/home/brian/Workspace/AUDIO_FINAL/data:/data", "audio", "python3", "scripts/batch_mix.py", f"/tmp/dig_mix_pi_{pert_indx}_l_{loudness}_config.json"])
