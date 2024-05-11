import os
import subprocess
from PIL import Image

def face_upscale_codeformer(latest_image):
    os.makedirs('/content/GenX/images/inputs', exist_ok=True)
    os.makedirs('/content/GenX/images/results', exist_ok=True)

    subprocess.run(['python', '/content/GenX/repository/CodeFormer/inference_codeformer.py',
                    '--input_path', '/content/GenX/images/inputs/',
                    '--output_path', '/content/GenX/images/results/',
                    '--face_upsample',
                    '-w', '0.7'])

    # Get the latest file in the final_results directory
    final_results_dir = '/content/GenX/images/results/final_results'
    latest_file = max([os.path.join(final_results_dir, f) for f in os.listdir(final_results_dir)], key=os.path.getctime)

    # Open the latest image using PIL
    latest_image = Image.open(latest_file)
    return latest_image
