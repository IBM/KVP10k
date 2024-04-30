import os

def count_lines_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        # Only count lines that are not empty (excluding whitespace)
        return sum(1 for line in file if line.strip())

def count_python_lines(directory):
    total_lines = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.py'):
                total_lines += count_lines_in_file(os.path.join(root, file))
    return total_lines

# Use the path to your PyCharm project directory here
project_directory = '.'
total_lines = count_python_lines(project_directory)
print(f'Total lines of Python code: {total_lines}')
