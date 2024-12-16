# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "pandas",
#   "matplotlib",
#   "seaborn",
#   "openai",
#   "tenacity"
# ]
# ///
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import requests

# Function to load data from a CSV file
def load_data(file_path):
    """Load data from a CSV file."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')

# Function for extended analysis on the dataset
def extended_analysis(df):
    """Perform extended analysis on the dataset."""
    analysis = {}
    analysis['shape'] = df.shape
    analysis['columns'] = df.columns.tolist()
    analysis['dtypes'] = df.dtypes.apply(lambda x: str(x)).tolist()
    analysis['summary'] = df.describe(include='all').to_dict()
    analysis['missing_values'] = df.isnull().sum().to_dict()

    # Detect columns with missing values
    analysis['missing_value_columns'] = [col for col, val in analysis['missing_values'].items() if val > 0]
    
    # Outlier detection using IQR for numeric columns
    numeric_cols = df.select_dtypes(include='number').columns
    analysis['outliers'] = {}
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outlier_count = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        analysis['outliers'][col] = outlier_count

    return analysis

# Function to generate up to 3 visualizations and save them as PNG files
def generate_plots(df, output_dir):
    """Generate up to 3 visualizations and save them as PNG files."""
    sns.set(style="darkgrid")
    image_files = []
    generated_count = 0

    # Distribution plot for a numerical column
    numeric_cols = df.select_dtypes(include='number').columns
    if numeric_cols.any():
        plt.figure(figsize=(10, 6))
        sns.histplot(df[numeric_cols[0]], kde=True)
        plt.title(f'Distribution of {numeric_cols[0]}')
        image_file = os.path.join(output_dir, f'{numeric_cols[0]}_distribution.png')
        plt.savefig(image_file)
        image_files.append(image_file)
        plt.close()
        generated_count += 1

    # Boxplot for the second numerical column
    if len(numeric_cols) > 1 and generated_count < 3:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=df[numeric_cols[1]])
        plt.title(f'Boxplot of {numeric_cols[1]}')
        image_file = os.path.join(output_dir, f'{numeric_cols[1]}_boxplot.png')
        plt.savefig(image_file)
        image_files.append(image_file)
        plt.close()
        generated_count += 1

    # Correlation heatmap
    if len(numeric_cols) > 1 and generated_count < 3:
        corr = df[numeric_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        image_file = os.path.join(output_dir, 'correlation_matrix.png')
        plt.savefig(image_file)
        image_files.append(image_file)
        plt.close()
        generated_count += 1

    return image_files

# Function to construct a concise dynamic LLM prompt
def construct_dynamic_prompt(analysis):
    """Construct a concise LLM prompt based on dataset characteristics."""
    prompt = (
        f"Dataset Analysis:\n"
        f"- Shape: {analysis['shape']}\n"
        f"- Columns: {', '.join(analysis['columns'])}\n"
        f"- Missing Value Columns: {', '.join(analysis['missing_value_columns']) if analysis['missing_value_columns'] else 'None'}\n"
        f"- Columns with Outliers: {', '.join([col for col, cnt in analysis['outliers'].items() if cnt > 0])}"
    )
    return prompt

# Function to query the LLM for insights
def query_llm(prompt):
    """Query the LLM with a concise prompt."""
    aiproxy_token = os.getenv("eyJhbGciOiJIUzI1NiJ9.eyJlbWFpbCI6IjIyZjMwMDE2NTJAZHMuc3R1ZHkuaWl0bS5hYy5pbiJ9.6Awo3wRrJsUNnYb5ExJuXDn0QfrsZ7uhTCjp6ILYsyA")
    if not aiproxy_token:
        raise EnvironmentError("AIPROXY_TOKEN not found in environment variables.")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {aiproxy_token}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}]
    }
    response = requests.post(
        "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    if response.status_code != 200:
        raise Exception(f"Error querying LLM: {response.status_code} {response.text}")

    response_json = response.json()
    if 'choices' not in response_json:
        raise KeyError(f"Unexpected response format: {response_json}")

    return response_json['choices'][0]['message']['content'].strip()

# Function to create the README.md with analysis story and images
def create_readme(analysis, story, image_files, output_dir):
    """Create a README.md file with the analysis story."""
    readme_path = os.path.join(output_dir, "README.md")
    with open(readme_path, "w") as f:
        f.write("# Automated Data Analysis Report\n\n")
        f.write("## Analysis Summary\n\n")
        f.write(f"- Shape of the dataset: {analysis['shape']}\n")
        f.write(f"- Columns with Missing Values: {', '.join(analysis['missing_value_columns']) if analysis['missing_value_columns'] else 'None'}\n")
        f.write(f"- Columns with Outliers: {', '.join([col for col, cnt in analysis['outliers'].items() if cnt > 0])}\n\n")
        f.write("## Analysis Story\n\n")
        f.write(story)
        f.write("\n\n## Visualizations\n\n")
        for img in image_files:
            f.write(f"![{img}](./{img})\n")

# Main function to run the analysis
def main():
    if len(sys.argv) != 2:
        print("Usage: python autolysis.py <path_to_csv_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    if not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)

    output_dir = os.path.splitext(os.path.basename(file_path))[0]
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(file_path)
    analysis = extended_analysis(df)
    image_files = generate_plots(df, output_dir)

    prompt = construct_dynamic_prompt(analysis)
    try:
        story = query_llm(prompt)
    except Exception as e:
        story = f"Failed to query LLM: {e}"

    create_readme(analysis, story, image_files, output_dir)
    print("Analysis completed. Check the output directory for results.")

if __name__ == "__main__":
    main()