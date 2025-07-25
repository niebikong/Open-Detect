import os
from tqdm import tqdm
from utils import read_pcap_list
import json
from PIL import Image
import numpy as np
import multiprocessing as mp
from functools import partial
import time

def process_file(pcap_dir, image_dir, if_augment, flow_dir_name, pcap_filename):
    try:
        full_path = os.path.join(pcap_dir, flow_dir_name, pcap_filename)
        base_name = os.path.splitext(pcap_filename)[0]
        
        if not if_augment:
            image_path = os.path.join(image_dir, flow_dir_name, f"{base_name}.png")
            res = read_pcap_list(full_path)[0]
            flow_array = res.pop("data")
            img = Image.fromarray(flow_array.reshape(32, 32).astype(np.uint8))
            img.save(image_path)
            return True
            
        else:  # Data augmentation mode
            res_list = read_pcap_list(full_path, if_augment=True)
            for i, res in enumerate(res_list):
                image_path = os.path.join(image_dir, flow_dir_name, f"{base_name}-{i}.png")
                stat_path = image_path.replace(".png", ".json")
                flow_array = res.pop("data")
                img = Image.fromarray(flow_array.reshape(32, 32).astype(np.uint8))
                img.save(image_path)
                with open(stat_path, "w") as f:
                    json.dump(res, f)
            return True
            
    except Exception as e:
        print(f"Error processing {flow_dir_name}/{pcap_filename}: {e}")
        return False

def pcap_to_array_parallel(pcap_dir, if_augment=False):
    assert os.path.basename(pcap_dir) == "malicious_TLS_4_paper"
    image_dir = pcap_dir.replace("malicious_TLS_4_paper", "pngs")
    
    # Create output directory structure
    flow_dir_names = [d for d in os.listdir(pcap_dir) if os.path.isdir(os.path.join(pcap_dir, d))]
    os.makedirs(image_dir, exist_ok=True)
    for d in flow_dir_names:
        os.makedirs(os.path.join(image_dir, d), exist_ok=True)

    # Prepare task list
    tasks = []
    for flow_dir in flow_dir_names:
        flow_dir_path = os.path.join(pcap_dir, flow_dir)
        pcap_files = [f for f in os.listdir(flow_dir_path) if f.endswith('.pcap')]
        for pcap_file in pcap_files:
            tasks.append((flow_dir, pcap_file))
    
    # Create worker function with fixed parameters
    worker = partial(process_file, pcap_dir, image_dir, if_augment)
    
    # Multiprocessing
    total_files = len(tasks)
    print(f"Starting processing of {total_files} files using {mp.cpu_count()} processes...")
    
    results = []
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # Use starmap to handle tuple arguments
        for result in tqdm(pool.starmap(worker, tasks), total=total_files, desc="Processing PCAPs", unit="file"):
            results.append(result)
    
    success_count = sum(results)
    failed_count = total_files - success_count
    
    print(f"\nProcessing completed. Success: {success_count}, Failed: {failed_count}")
    return success_count

if __name__ == "__main__":
    start = time.time()
    pcap_dir = "/dataset/raw_pcap/malicious_TLS_4_paper"
    if_augment = False
    
    success_count = pcap_to_array_parallel(pcap_dir, if_augment)
    duration = time.time() - start
    
    if success_count > 0:
        print(f"Total time: {duration:.2f} seconds")
        print(f"Processing speed: {success_count/duration:.2f} files/second")
    