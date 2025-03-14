#!/bin/bash

# Directory containing PDF files
input_dir="./mqss-passes"
output_dir="./mqss-passes"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through each PDF file in the input directory
for pdf_file in "$input_dir"/*.pdf; do
  # Extract the base name (file name without extension)
  base_name=$(basename "$pdf_file" .pdf)

  # Convert the PDF to PNG using ImageMagick
  convert -density 300 "$pdf_file" -quality 100 "$output_dir/$base_name.png"

  echo "Converted $pdf_file to $output_dir/$base_name.png"
done
