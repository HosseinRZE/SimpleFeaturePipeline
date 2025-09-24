
import psutil
def resource_usage():
    """Print current CPU, RAM, and GPU usage."""
    cpu = psutil.cpu_percent(interval=0.5)
    ram = psutil.virtual_memory().percent
    usage = f"💻 CPU: {cpu:.1f}% | 🧠 RAM: {ram:.1f}%"
    try:
        import GPUtil
        gpus = GPUtil.getGPUs()
        if gpus:
            usage += f" | 🎮 GPU: {gpus[0].load*100:.1f}% VRAM: {gpus[0].memoryUtil*100:.1f}%"
    except ImportError:
        pass
    print(usage)
