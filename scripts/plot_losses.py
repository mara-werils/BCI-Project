import argparse
import matplotlib.pyplot as plt
import re
import os

def parse_log(log_path):
    parsed_data = []
    keys_found = set()
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line.startswith('('):
                continue
            
            # Parse metadata (epoch, iters)
            meta_match = re.search(r'\(epoch:\s*(\d+),\s*iters:\s*(\d+)', line)
            if not meta_match:
                continue
            
            # Extract the part after the metadata
            try:
                content = line.split(') ')[1]
            except IndexError:
                continue
                
            parts = content.split()
            item = {}
            valid_line = True
            
            # Key-value pairs like "G_GAN: 1.234"
            current_key = None
            for part in parts:
                if ':' in part:
                    current_key = part.replace(':', '')
                    keys_found.add(current_key)
                elif current_key:
                    try:
                        val = float(part)
                        item[current_key] = val
                        current_key = None
                    except ValueError:
                        pass
            
            if item:
                item['epoch'] = int(meta_match.group(1))
                item['iters'] = int(meta_match.group(2))
                parsed_data.append(item)
                
    return parsed_data, list(keys_found)

def plot_losses(data, keys, output_path):
    if not data:
        print("No data found to plot.")
        return

    plt.figure(figsize=(12, 6))
    
    # Create x-axis (cumulative iterations or just index)
    x = range(len(data))
    
    for key in keys:
        y = [d.get(key, 0) for d in data] # Use 0 or previous value if missing? 0 is safer for now
        if any(y): # Only plot if there's data
            plt.plot(x, y, label=key)
        
    plt.xlabel('Log Entry (Time)')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.grid(True)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"Loss plot saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot losses from Pix2Pix training log.')
    parser.add_argument('--log_path', type=str, required=True, help='Path to loss_log.txt')
    parser.add_argument('--output_path', type=str, default='results/loss_plot.png', help='Path to save the plot')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_path):
        print(f"Error: Log file not found at {args.log_path}")
        exit(1)
        
    data, keys = parse_log(args.log_path)
    plot_losses(data, keys, args.output_path)
