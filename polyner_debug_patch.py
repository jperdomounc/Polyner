"""
Debug patch for Polyner.py to add iteration monitoring and debugging
This script adds debugging hooks without modifying the core algorithm
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

class PolynerIterationMonitor:
    """Monitor Polyner training iterations and save debugging data"""
    
    def __init__(self, output_dir="./debug_output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.iteration_data = []
        self.proj_data = []
        self.intensity_data = []
        
        print(f"🔍 Polyner Debug Monitor initialized - output: {output_dir}")
    
    def log_iteration(self, epoch, loss, p_hat=None, proj_pre=None, intensity_pre=None):
        """Log data from each iteration"""
        
        iteration_info = {
            'epoch': epoch,
            'loss': float(loss.item()) if torch.is_tensor(loss) else float(loss),
            'timestamp': datetime.now().isoformat()
        }
        
        if p_hat is not None:
            if torch.is_tensor(p_hat):
                p_stats = {
                    'p_mean': float(p_hat.mean()),
                    'p_std': float(p_hat.std()),
                    'p_min': float(p_hat.min()),
                    'p_max': float(p_hat.max()),
                    'p_shape': list(p_hat.shape)
                }
            else:
                p_stats = {
                    'p_mean': float(np.mean(p_hat)),
                    'p_std': float(np.std(p_hat)),
                    'p_min': float(np.min(p_hat)),
                    'p_max': float(np.max(p_hat))
                }
            iteration_info.update(p_stats)
        
        self.iteration_data.append(iteration_info)
        
        # Store projection and intensity data periodically
        if epoch % 100 == 0 or epoch < 10:
            if proj_pre is not None:
                proj_array = proj_pre.detach().cpu().numpy() if torch.is_tensor(proj_pre) else proj_pre
                self.proj_data.append({
                    'epoch': epoch,
                    'data': proj_array,
                    'shape': proj_array.shape,
                    'mean': np.mean(proj_array),
                    'std': np.std(proj_array)
                })
            
            if intensity_pre is not None:
                intensity_array = intensity_pre.detach().cpu().numpy() if torch.is_tensor(intensity_pre) else intensity_pre
                self.intensity_data.append({
                    'epoch': epoch,
                    'data': intensity_array,
                    'shape': intensity_array.shape,
                    'mean': np.mean(intensity_array),
                    'std': np.std(intensity_array)
                })
        
        # Print progress
        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss={iteration_info['loss']:.6f}", end="")
            if 'p_mean' in iteration_info:
                print(f", P_mean={iteration_info['p_mean']:.6f}, P_std={iteration_info['p_std']:.6f}")
            else:
                print()
    
    def save_iteration_plots(self):
        """Create plots showing iteration progress"""
        if not self.iteration_data:
            print("No iteration data to plot")
            return
        
        epochs = [d['epoch'] for d in self.iteration_data]
        losses = [d['loss'] for d in self.iteration_data]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curve
        axes[0,0].plot(epochs, losses, 'b-', linewidth=1)
        axes[0,0].set_xlabel('Epoch')
        axes[0,0].set_ylabel('Loss')
        axes[0,0].set_title('Training Loss')
        axes[0,0].grid(True, alpha=0.3)
        
        # P value statistics
        if 'p_mean' in self.iteration_data[0]:
            p_means = [d['p_mean'] for d in self.iteration_data if 'p_mean' in d]
            p_stds = [d['p_std'] for d in self.iteration_data if 'p_std' in d]
            p_epochs = [d['epoch'] for d in self.iteration_data if 'p_mean' in d]
            
            axes[0,1].plot(p_epochs, p_means, 'r-', label='Mean', linewidth=1)
            axes[0,1].fill_between(p_epochs, 
                                 [m-s for m,s in zip(p_means, p_stds)], 
                                 [m+s for m,s in zip(p_means, p_stds)], 
                                 alpha=0.3)
            axes[0,1].set_xlabel('Epoch')
            axes[0,1].set_ylabel('P Value')
            axes[0,1].set_title('P Value Statistics')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Projection statistics
        if self.proj_data:
            proj_epochs = [d['epoch'] for d in self.proj_data]
            proj_means = [d['mean'] for d in self.proj_data]
            proj_stds = [d['std'] for d in self.proj_data]
            
            axes[1,0].plot(proj_epochs, proj_means, 'g-', linewidth=2, label='Mean')
            axes[1,0].plot(proj_epochs, proj_stds, 'g--', linewidth=1, label='Std')
            axes[1,0].set_xlabel('Epoch')
            axes[1,0].set_ylabel('Projection Value')
            axes[1,0].set_title('Projection Statistics')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Intensity statistics
        if self.intensity_data:
            int_epochs = [d['epoch'] for d in self.intensity_data]
            int_means = [d['mean'] for d in self.intensity_data]
            int_stds = [d['std'] for d in self.intensity_data]
            
            axes[1,1].plot(int_epochs, int_means, 'm-', linewidth=2, label='Mean')
            axes[1,1].plot(int_epochs, int_stds, 'm--', linewidth=1, label='Std')
            axes[1,1].set_xlabel('Epoch')
            axes[1,1].set_ylabel('Intensity Value')
            axes[1,1].set_title('Intensity Statistics')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = os.path.join(self.output_dir, 'iteration_analysis.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.show()
        print(f"✓ Iteration plots saved to {plot_path}")
    
    def save_debug_data(self):
        """Save all debug data to files"""
        import json
        
        # Save iteration data as JSON
        json_path = os.path.join(self.output_dir, 'iteration_data.json')
        with open(json_path, 'w') as f:
            json.dump(self.iteration_data, f, indent=2)
        
        # Save projection data
        if self.proj_data:
            proj_path = os.path.join(self.output_dir, 'projection_data.npz')
            proj_arrays = {f'epoch_{d["epoch"]}': d['data'] for d in self.proj_data}
            np.savez(proj_path, **proj_arrays)
            print(f"✓ Projection data saved to {proj_path}")
        
        # Save intensity data
        if self.intensity_data:
            int_path = os.path.join(self.output_dir, 'intensity_data.npz')
            int_arrays = {f'epoch_{d["epoch"]}': d['data'] for d in self.intensity_data}
            np.savez(int_path, **int_arrays)
            print(f"✓ Intensity data saved to {int_path}")
        
        print(f"✓ Iteration data saved to {json_path}")
    
    def generate_report(self):
        """Generate debugging report"""
        if not self.iteration_data:
            print("No data to generate report")
            return
        
        report = f"""
=== POLYNER DEBUGGING REPORT ===
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

TRAINING STATISTICS:
- Total epochs: {len(self.iteration_data)}
- Final loss: {self.iteration_data[-1]['loss']:.6f}
- Initial loss: {self.iteration_data[0]['loss']:.6f}
- Loss reduction: {((self.iteration_data[0]['loss'] - self.iteration_data[-1]['loss']) / self.iteration_data[0]['loss'] * 100):.2f}%

"""
        
        if 'p_mean' in self.iteration_data[-1]:
            report += f"""P VALUE ANALYSIS:
- Final P mean: {self.iteration_data[-1]['p_mean']:.6f}
- Final P std: {self.iteration_data[-1]['p_std']:.6f}
- P range: [{self.iteration_data[-1]['p_min']:.6f}, {self.iteration_data[-1]['p_max']:.6f}]

"""
        
        if self.proj_data:
            report += f"""PROJECTION DATA:
- Stored {len(self.proj_data)} snapshots
- Projection shape: {self.proj_data[-1]['shape']}
- Final proj mean: {self.proj_data[-1]['mean']:.6f}
- Final proj std: {self.proj_data[-1]['std']:.6f}

"""
        
        if self.intensity_data:
            report += f"""INTENSITY DATA:
- Stored {len(self.intensity_data)} snapshots
- Intensity shape: {self.intensity_data[-1]['shape']}
- Final intensity mean: {self.intensity_data[-1]['mean']:.6f}
- Final intensity std: {self.intensity_data[-1]['std']:.6f}

"""
        
        report_path = os.path.join(self.output_dir, 'debug_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        
        print(report)
        print(f"✓ Full report saved to {report_path}")

# Global monitor instance
monitor = None

def init_monitor(output_dir="./debug_output"):
    """Initialize the global monitor"""
    global monitor
    monitor = PolynerIterationMonitor(output_dir)
    return monitor

def log_iteration(epoch, loss, p_hat=None, proj_pre=None, intensity_pre=None):
    """Log iteration data (call this from your training loop)"""
    global monitor
    if monitor is None:
        monitor = init_monitor()
    monitor.log_iteration(epoch, loss, p_hat, proj_pre, intensity_pre)

def finalize_monitoring():
    """Call this at the end of training"""
    global monitor
    if monitor is not None:
        monitor.save_iteration_plots()
        monitor.save_debug_data()
        monitor.generate_report()

if __name__ == "__main__":
    print("Polyner Debug Patch - Use init_monitor() in your training script")
    print("Example usage:")
    print("  from polyner_debug_patch import init_monitor, log_iteration, finalize_monitoring")
    print("  monitor = init_monitor()")
    print("  # In training loop:")
    print("  log_iteration(epoch, loss, p_hat, proj_pre, intensity_pre)")
    print("  # After training:")
    print("  finalize_monitoring()")