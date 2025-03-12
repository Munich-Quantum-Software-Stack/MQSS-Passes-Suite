import os
import sys
import shutil

def merge_files(file1_path, file2_path, output_path):
    """Merges two files and writes the merged content into the output file."""
    try:
        with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2, open(output_path, 'w') as output_file:
            # Read and write content from both files into the output file
            content1 = file1.read()
            content2 = file2.read()
            output_file.write("\\documentclass[border=6pt]{standalone}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n")
            output_file.write("\\usepackage{tikz}\n\\usetikzlibrary{quantikz2}\n")
            output_file.write("\\begin{document}")
            output_file.write(content1 + "\n{\\Huge{\\textbf{=}}}\n" + content2+"\n")  # Combine content of both files
            output_file.write("\\end{document}")
        print(f"Merged {file1_path} and {file2_path} into {output_path}")
    except Exception as e:
        print(f"Error merging files {file1_path} and {file2_path}: {e}")

def merge_folders(input_dir1, input_dir2, output_dir):
    """Merges files with the same name in two directories and saves the result to an output directory."""
    # Check if the input directories exist
    if not os.path.isdir(input_dir1) or not os.path.isdir(input_dir2):
        print("Both input directories must exist.")
        sys.exit(1)

    # Check if the output directory exists, create it if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get lists of files in both directories
    files_dir1 = set(os.listdir(input_dir1))
    files_dir2 = set(os.listdir(input_dir2))

    # Find common files between both directories
    common_files = files_dir1.intersection(files_dir2)

    # Iterate over common files and merge them
    for file_name in common_files:
        file1_path = os.path.join(input_dir1, file_name)
        file2_path = os.path.join(input_dir2, file_name)
        output_path = os.path.join(output_dir, file_name)
        merge_files(file1_path, file2_path, output_path)

def main():
    # Check if enough arguments are passed
    if len(sys.argv) != 4:
        print("Usage: python3 merge_folders.py <input_folder1> <input_folder2> <output_folder>")
        sys.exit(1)

    input_dir1 = sys.argv[1]
    input_dir2 = sys.argv[2]
    output_dir = sys.argv[3]

    # Call the function to merge the files
    merge_folders(input_dir1, input_dir2, output_dir)

if __name__ == "__main__":
    main()
