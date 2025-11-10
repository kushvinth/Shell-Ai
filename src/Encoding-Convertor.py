import chardet # type: ignore

def detect_encoding(file_path, num_bytes=10000):
    with open(file_path, 'rb') as f:
        raw = f.read(num_bytes)
    result = chardet.detect(raw)
    return result['encoding']

def convert_to_utf8(input_file, output_file):
    original_encoding = detect_encoding(input_file)
    print(f"Detected encoding: {original_encoding}")
    
    with open(input_file, 'r', encoding=original_encoding, errors='ignore') as src:
        content = src.read()
    
    with open(output_file, 'w', encoding='utf-8') as dest:
        dest.write(content)
    
    print(f"Converted file saved to: {output_file}")

# Example usage:
input_csv = '/Users/MacbookPro/Downloads/TRAIL1.csv'
output_csv = 'finn.csv'
convert_to_utf8(input_csv, output_csv)


