import subprocess

scripts = [
    "eda.py",
    "feature_engineering.py",
    "model_training.py",
    "scheduling.py",
    "forecast_output.py",
    "siteintel.py"
]

for script in scripts:
    print(f"Running {script}...")
    subprocess.run(["python", script], check=True)
    print(f"{script} done.\n")