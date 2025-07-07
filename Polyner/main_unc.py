# ----------------------------------------------#
# Pro    : cbct
# File   : main_unc.py  
# Date   : 2025/7/2
# Author : Claude Code Assistant
# Email  : Modified for UNC test data
# ----------------------------------------------#
import Polyner
import commentjson as json
import sys
import os

if __name__ == '__main__':

    # load config
    # -----------------------
    config_file = "config_unc.json" if len(sys.argv) < 2 else sys.argv[1]
    
    print(f"Loading configuration from: {config_file}")
    with open(config_file) as f:
        config = json.load(f)

    # Create output directories
    os.makedirs(config["file"]["model_dir"], exist_ok=True)
    os.makedirs(config["file"]["out_dir"], exist_ok=True)

    # train on UNC test cases
    # -----------------------
    num_cases = 3  # We prepared 3 test cases
    
    print(f"Starting Polyner training on {num_cases} UNC test cases...")
    
    for i in range(num_cases):
        print(f"\n{'='*50}")
        print(f"Training case {i} (slice {55 + i*5})")
        print(f"{'='*50}")
        
        try:
            Polyner.train(img_id=i, config=config)
            print(f"✓ Case {i} completed successfully")
        except Exception as e:
            print(f"✗ Error in case {i}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print("UNC Polyner training completed!")
    print(f"Check results in: {config['file']['out_dir']}")
    print(f"{'='*50}")