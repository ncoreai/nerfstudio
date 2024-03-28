import os

def generate_markdown_file(repo_path, output_file):
    with open(output_file, 'w') as f:
        for root, dirs, files in os.walk(repo_path):
            for file in files:
                if file.endswith('.py'):
                    print(f"Processing {file}")
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, repo_path)
                    directory = os.path.dirname(relative_path)

                    # Write the directory path as a Markdown heading
                    f.write(f"## {directory}\n\n")

                    # Write the file name as a Markdown subheading
                    f.write(f"### {file}\n\n")

                    # Write the contents of the script
                    with open(file_path, 'r') as script:
                        f.write("```python\n")
                        f.write(script.read())
                        f.write("\n```\n\n")

# Example usage
repo_path = '/temp/pynerf'
output_file = '/workspace/pynerf_code.md'

generate_markdown_file(repo_path, output_file)