import os
import subprocess
import sys
import argparse

def convert_cpp_to_tikz(tool, input_dir, output_dir):
  # Check if the input directory exists
  if not os.path.isdir(input_dir):
    print(f"The directory {input_dir} does not exist.")
    sys.exit(1)

  # Check if the output directory exists, create it if not
  if not os.path.exists(output_dir):
    print(f"The output directory {output_dir} does not exist. Creating it now.")
    os.makedirs(output_dir)

  # Loop over all .cpp files in the input directory
  for file_name in os.listdir(input_dir):
    # Check if the file has a .cpp extension
    if file_name.endswith(".cpp") or file_name.endswith(".qke"):
      # Full path to the input file
      cpp_file = os.path.join(input_dir, file_name)

      # Create output file name by replacing .cpp with .tikz
      tikz_file = os.path.join(output_dir, file_name.rsplit('.', 1)[0] + ".tikz")
      # Run the quake-to-tikz command
      try:
        subprocess.run([tool, '--input', cpp_file, '--output', tikz_file], check=True)
        print(f"Generated {tikz_file} from {cpp_file}")
      except subprocess.CalledProcessError as e:
        print(f"Error while processing {cpp_file}: {e}")

def main():
  # Set up argparse for command-line arguments
  parser = argparse.ArgumentParser(description="Convert .cpp files to .tikz files using quake-to-tikz.")

  # Define the argument to specify the dumper tool
  parser.add_argument('tool', metavar='tool_dir', type=str, 
                      help="The tool used to dump each cpp or qke file")

  # Define the arguments for input and output directories
  parser.add_argument('input_dir', metavar='input_directory', type=str, 
                      help="Directory containing the .cpp files to convert. This argument is mandatory.")
  parser.add_argument('output_dir', metavar='output_directory', type=str, 
                      help="Directory to store the generated .tikz files. This argument is mandatory.")

  # Parse the arguments
  args = parser.parse_args()

  # Check if both input and output directories are provided
  if not args.input_dir or not args.output_dir:
    print("Both input directory and output directory are mandatory.")
    sys.exit(1)

  # Call the function to perform the conversion
  convert_cpp_to_tikz(args.tool, args.input_dir, args.output_dir)

if __name__ == "__main__":
  main()

