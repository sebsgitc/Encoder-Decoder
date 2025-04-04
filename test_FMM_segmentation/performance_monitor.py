"""Performance monitoring utilities for deep learning training."""

import os
import time
import threading
import subprocess
import psutil
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

class TrainingPerformanceMonitor:
    """Monitor GPU and CPU usage during training."""
    
    def __init__(self, log_interval=5, output_dir="performance_logs"):
        """Initialize the performance monitor."""
        self.log_interval = log_interval
        self.output_dir = output_dir
        self.running = False
        self.monitor_thread = None
        self.gpu_utilization = []
        self.gpu_memory = []
        self.cpu_utilization = []
        self.ram_usage = []
        self.timestamps = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def start(self):
        """Start monitoring performance."""
        if self.running:
            print("Performance monitor is already running.")
            return
            
        self.running = True
        self.gpu_utilization = []
        self.gpu_memory = []
        self.cpu_utilization = []
        self.ram_usage = []
        self.timestamps = []
        self.start_time = time.time()
        
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
        print(f"Performance monitoring started. Logging every {self.log_interval} seconds.")
        
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            # Get current timestamp
            current_time = time.time() - self.start_time
            self.timestamps.append(current_time)
            
            # Get GPU stats using nvidia-smi
            try:
                gpu_stats = self._get_gpu_stats()
                self.gpu_utilization.append(gpu_stats['utilization'])
                self.gpu_memory.append(gpu_stats['memory'])
            except Exception as e:
                print(f"Error getting GPU stats: {e}")
                self.gpu_utilization.append(0)
                self.gpu_memory.append(0)
            
            # Get CPU stats
            cpu_percent = psutil.cpu_percent()
            self.cpu_utilization.append(cpu_percent)
            
            # Get RAM usage
            ram_percent = psutil.virtual_memory().percent
            self.ram_usage.append(ram_percent)
            
            # Sleep for the logging interval
            time.sleep(self.log_interval)
    
    def _get_gpu_stats(self):
        """Get GPU utilization and memory usage."""
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True
        )
        
        # Parse output
        utilization, memory = map(float, result.stdout.strip().split(','))
        return {'utilization': utilization, 'memory': memory}
    
    def stop(self):
        """Stop monitoring and generate report."""
        if not self.running:
            print("Performance monitor is not running.")
            return
            
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        # Generate report
        self._generate_report()
        
        print("Performance monitoring stopped and report generated.")
    
    def _generate_report(self):
        """Generate performance report."""
        if not self.timestamps:
            print("No data collected. Cannot generate report.")
            return
        
        # Create plots
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        
        # GPU plot
        axs[0].plot(self.timestamps, self.gpu_utilization, 'r-', label='GPU Utilization (%)')
        axs[0].set_xlabel('Time (seconds)')
        axs[0].set_ylabel('GPU Utilization (%)')
        axs[0].set_title('GPU Performance')
        axs[0].grid(True)
        
        # Add GPU memory as a secondary axis
        ax2 = axs[0].twinx()
        ax2.plot(self.timestamps, self.gpu_memory, 'b-', label='GPU Memory (MB)')
        ax2.set_ylabel('GPU Memory (MB)')
        
        # Combine legends
        lines1, labels1 = axs[0].get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        axs[0].legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # CPU/RAM plot
        axs[1].plot(self.timestamps, self.cpu_utilization, 'g-', label='CPU Utilization (%)')
        axs[1].plot(self.timestamps, self.ram_usage, 'm-', label='RAM Usage (%)')
        axs[1].set_xlabel('Time (seconds)')
        axs[1].set_ylabel('Utilization (%)')
        axs[1].set_title('CPU and RAM Performance')
        axs[1].grid(True)
        axs[1].legend()
        
        # Save plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f'performance_report_{timestamp}.png'))
        plt.close(fig)
        
        # Save raw data as CSV
        data = np.column_stack((
            self.timestamps, 
            self.gpu_utilization, 
            self.gpu_memory, 
            self.cpu_utilization, 
            self.ram_usage
        ))
        
        header = "Time(s),GPU_Util(%),GPU_Mem(MB),CPU_Util(%),RAM_Usage(%)"
        np.savetxt(
            os.path.join(self.output_dir, f'performance_data_{timestamp}.csv'),
            data,
            delimiter=',',
            header=header,
            comments=''
        )
        
        # Print summary statistics
        print("\nPerformance Summary:")
        print(f"Duration: {self.timestamps[-1]:.2f} seconds")
        print(f"Average GPU Utilization: {np.mean(self.gpu_utilization):.2f}%")
        print(f"Average GPU Memory Usage: {np.mean(self.gpu_memory):.2f} MB")
        print(f"Average CPU Utilization: {np.mean(self.cpu_utilization):.2f}%")
        print(f"Average RAM Usage: {np.mean(self.ram_usage):.2f}%")

# Example usage
if __name__ == "__main__":
    monitor = TrainingPerformanceMonitor(log_interval=1)
    
    print("Starting performance monitoring demo...")
    monitor.start()
    
    try:
        # Simulate training for 30 seconds
        for i in range(30):
            print(f"Training iteration {i+1}/30")
            time.sleep(1)
    finally:
        monitor.stop()
