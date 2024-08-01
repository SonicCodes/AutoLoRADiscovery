# from train_utils import default_arguments
# from train_lora import train
# import os
# import glob
# import sys
# import subprocess
# if __name__ == "__main__":
#     lora_download_folder = "/mnt/rd/lora_outputs_2"
#     people_folder = "/mnt/rd/celebpic/cropped"
#     # people = [os.path.join(people_folder,f) for f in os.listdir(people_folder)]
#     os.makedirs(lora_download_folder, exist_ok=True)
#     # square_images = torch.load("/home/ubuntu/AutoLoRADiscovery/square_images.pth")
#     # unique_celeb = set([image.split('/')[-2] for image in square_images])
#     unique_celeb = os.listdir(people_folder)
#     people = [os.path.join(people_folder,celeb) for celeb in unique_celeb if len(glob.glob(os.path.join(people_folder,celeb)+"/*.jpg")) >= 40]

#     print("Training loras for ", len(people))
#     for person in people:
        
#         # skip if there are no .jpg here
#         if len(glob.glob(person+"/*.jpg")) < 40:
#             continue
#         if os.path.exists(lora_download_folder+"/lora-"+person.split("/")[-1]):
#             print ("Already trained for ", person.split("/")[-1])
#             continue
#         # raise Exception("Training for ", person.split("/")[-1])
#         # call ./train_lora.py with the person's name and details
#         ls_output= subprocess.Popen(["python3", "train_lora.py", "--celeb_name", person.split("/")[-1], "--instance_data_dir", person, "--output_dir", lora_download_folder+"/lora-"+person.split("/")[-1]])
#         ls_output.communicate()
#         # args = default_arguments
#         # args["celeb_name"] = person.split("/")[-1]
#         # args["instance_data_dir"] = person
#         # args["output_dir"] = lora_download_folder+"/lora-"+person.split("/")[-1]
#         # train(args)

import os
import glob
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def train_lora(person, lora_download_folder):
    celeb_name = person.split("/")[-1]
    if os.path.exists(f"{lora_download_folder}/lora-{celeb_name}"):
        print(f"Already trained for {celeb_name}")
        return
    os.makedirs(f"{lora_download_folder}", exist_ok=True)
    try:
        with open(f"{lora_download_folder}/lora-{celeb_name}.log", "w") as log_file:
            process = subprocess.Popen(
                ["python3", "train_lora.py", "--celeb_name", celeb_name, "--instance_data_dir", person, "--output_dir", f"{lora_download_folder}/lora-{celeb_name}"],
                stdout=log_file,
                stderr=log_file
            )
            process.wait()
    except Exception as e:
        print(f"Training failed for {celeb_name}: {e}")

if __name__ == "__main__":
    lora_download_folder = "/mnt/rd/lora_outputs_2"
    people_folder = "/mnt/rd/celebpic/cropped"
    os.makedirs(lora_download_folder, exist_ok=True)
    
    unique_celeb = os.listdir(people_folder)
    people = [os.path.join(people_folder, celeb) for celeb in unique_celeb if len(glob.glob(os.path.join(people_folder, celeb) + "/*.jpg")) >= 40]

    print("Training loras for ", len(people))
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(train_lora, person, lora_download_folder) for person in people]
        
        with tqdm(total=len(futures)) as pbar:
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"An error occurred: {e}")
                pbar.update(1)
