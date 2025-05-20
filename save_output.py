import subprocess

def run_command(command, output_file):
    with open(output_file, 'w') as f:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
        for line in process.stdout:
            print(line, end='')  # Print to terminal
            f.write(line)        # Write to file

if __name__ == "__main__":
    command = "python run_experiments.py -e resnet50_ssal"
    output_file = "output.txt"
    run_command(command, output_file)
    print("Command execution completed. Output saved in", output_file)
