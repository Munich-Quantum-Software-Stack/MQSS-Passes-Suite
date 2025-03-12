import os
import subprocess
import sys

def compile_latex_to_pdf(tex_file, output_dir):
    """Compiles a LaTeX file to a PDF using pdflatex."""
    try:
        # Ensure output directory exists
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Run pdflatex
        output_pdf_dir = os.path.abspath(output_dir)
        tex_file_path = os.path.abspath(tex_file)
        compile_command = [
            "pdflatex",
            "-output-directory", output_pdf_dir,
            tex_file_path
        ]
        subprocess.run(compile_command, check=True)
        print(f"Successfully compiled {tex_file} to PDF.")
    except subprocess.CalledProcessError as e:
        print(f"Error compiling {tex_file}: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

def process_folder(input_dir, output_dir):
    """Processes all .tex files in a folder and generates PDFs."""
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory.")
        sys.exit(1)

    tex_files = [f for f in os.listdir(input_dir) if f.endswith(".tikz")]

    if not tex_files:
        print("No .tex files found in the specified directory.")
        return

    for tex_file in tex_files:
        tex_file_path = os.path.join(input_dir, tex_file)
        compile_latex_to_pdf(tex_file_path, output_dir)

def main():
    if len(sys.argv) != 3:
        print("Usage: python3 compile_latex.py <input_folder> <output_folder>")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    process_folder(input_dir, output_dir)

if __name__ == "__main__":
    main()
