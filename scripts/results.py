import os
import csv
import re

# Function to extract the relevant data from the log file content
def extract_metrics_from_log(log_file):
    # Define the regex pattern to match the desired line format
    pattern = r"Computed over (\d+) frames, (\d+) sentences - Frame-level accuracy: (\d+\.\d+) F1@0\.10: (\d+\.\d+) F1@0\.25: (\d+\.\d+) F1@0\.50: (\d+\.\d+)"
    dtw_start_pattern = r"after DTW output"
    dtw_pattern = r"Computed over (\d+) frames, (\d+) sentences - Frame-level accuracy: (\d+\.\d+) F1@0\.10: (\d+\.\d+) F1@0\.25: (\d+\.\d+) F1@0\.50: (\d+\.\d+)"

    frame_acc = f1_10 = f1_25 = f1_50 = ""
    frame_acc_dtw = f1_10_dtw = f1_25_dtw = f1_50_dtw = ""

    try:
        with open(log_file, 'r') as file:
            lines = file.readlines()

            # Find the first matching line (metrics before DTW)
            for line in lines:
                match = re.search(pattern, line)
                if match:
                    frame_acc = match.group(3)
                    f1_10 = match.group(4)
                    f1_25 = match.group(5)
                    f1_50 = match.group(6)
                    break

            # Find the line "after DTW output" and then look for the second matching line (metrics after DTW)
            after_dtw = False
            for line in lines:
                if re.search(dtw_start_pattern, line):
                    after_dtw = True
                if after_dtw:
                    match_dtw = re.search(dtw_pattern, line)
                    if match_dtw:
                        frame_acc_dtw = match_dtw.group(3)
                        f1_10_dtw = match_dtw.group(4)
                        f1_25_dtw = match_dtw.group(5)
                        f1_50_dtw = match_dtw.group(6)
                        break

    except Exception as e:
        print(f"Error processing file {log_file}: {e}")

    return frame_acc, f1_10, f1_25, f1_50, frame_acc_dtw, f1_10_dtw, f1_25_dtw, f1_50_dtw

# Main function to search the folders and log files, then write to CSV
def process_logs_to_csv(output_csv="results.csv"):
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.realpath(__file__))
    output_path = os.path.join(script_dir, output_csv)

    # Initialize the CSV file and write the headers
    with open(output_path, mode='w', newline='') as csvfile:
        fieldnames = ['method', 'frame-acc', 'F1@.10', 'F1@.25', 'F1@.50', 'frame-acc-dtw', 'F1@.10-dtw', 'F1@.25-dtw', 'F1@.50-dtw']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()

        # Define the specific order of folders to search
        folders = ['commands_reproduce', 'commands_swin', 'commands_pose', 'commands_pose_youtube']

        # Loop through the specific folder order
        for folder in folders:
            # Search for log files that start with 'test' and end with '.log' in the specific folder
            folder_path = os.path.join('.', folder)
            if os.path.isdir(folder_path):
                # First check for 'test.log' explicitly and process it first
                test_log_path = os.path.join(folder_path, 'test.log')
                if os.path.exists(test_log_path):
                    # Extract the metrics from 'test.log' first
                    frame_acc, f1_10, f1_25, f1_50, frame_acc_dtw, f1_10_dtw, f1_25_dtw, f1_50_dtw = extract_metrics_from_log(test_log_path)
                    
                    # Write the data to the CSV, removing the 'commands_' prefix from the folder name and '.log' suffix
                    writer.writerow({
                        'method': f'{folder.replace("commands_", "")}/test',
                        'frame-acc': frame_acc,
                        'F1@.10': f1_10,
                        'F1@.25': f1_25,
                        'F1@.50': f1_50,
                        'frame-acc-dtw': frame_acc_dtw,
                        'F1@.10-dtw': f1_10_dtw,
                        'F1@.25-dtw': f1_25_dtw,
                        'F1@.50-dtw': f1_50_dtw
                    })
                    print(f"Processed {folder}/test.log")

                # Now process the remaining test*.log files
                for file_name in os.listdir(folder_path):
                    if file_name.startswith("test") and file_name.endswith(".log") and file_name != "test.log":
                        log_file_path = os.path.join(folder_path, file_name)
                        
                        # Extract the metrics from the log file
                        frame_acc, f1_10, f1_25, f1_50, frame_acc_dtw, f1_10_dtw, f1_25_dtw, f1_50_dtw = extract_metrics_from_log(log_file_path)
                        
                        # Write the data to the CSV, removing the 'commands_' prefix from the folder name and '.log' suffix
                        writer.writerow({
                            'method': f'{folder.replace("commands_", "")}/{file_name.replace(".log", "")}',  # Remove the ".log" suffix
                            'frame-acc': frame_acc,
                            'F1@.10': f1_10,
                            'F1@.25': f1_25,
                            'F1@.50': f1_50,
                            'frame-acc-dtw': frame_acc_dtw,
                            'F1@.10-dtw': f1_10_dtw,
                            'F1@.25-dtw': f1_25_dtw,
                            'F1@.50-dtw': f1_50_dtw
                        })
                        print(f"Processed {folder}/{file_name}")

    print(f"Data extraction complete. Results saved to {output_path}")

# Run the main function
process_logs_to_csv()
