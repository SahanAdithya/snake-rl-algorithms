import subprocess
import sys
import os

def run_command(command):
    venv_python = os.path.join("venv", "bin", "python3")
    if not os.path.exists(venv_python):
        venv_python = "python3" # Fallback to system python if venv not found
        print("Warning: venv/bin/python3 not found, using system python3")
    
    full_command = [venv_python] + command
    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error: Command failed with exit code {e.returncode}")
    except KeyboardInterrupt:
        print("\nProcess stopped by user.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 run.py [train|test]")
        sys.exit(1)
    
    mode = sys.argv[1].lower()
    if mode == "train":
        run_command(["src/trainer.py"])
    elif mode == "test":
        run_command(["src/test_agent.py"])
    elif mode == "optimize":
        run_command(["src/optimize.py"])
    else:
        print(f"Unknown mode: {mode}. Use 'train' or 'test'.")
