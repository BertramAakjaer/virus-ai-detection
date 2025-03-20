import pefile, os, math

def parse_pe(file_path):
    pe = pefile.PE(file_path)
    
    print("DOS Header:")
    print(pe.DOS_HEADER)
    
    print("\nPE Header:")
    print(pe.NT_HEADERS)
    
    print("\nOptional Header:")
    print(pe.OPTIONAL_HEADER)
    
    print("\nSection Headers:")
    for section in pe.sections:
        print(section.Name.decode().strip(), hex(section.VirtualAddress), section.SizeOfRawData)
        
def enumerate_imports(file_path):
    pe = pefile.PE(file_path)
    
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            print(f"Library: {entry.dll.decode()}")
            for imp in entry.imports:
                print(f"\t{imp.name.decode() if imp.name else imp.ordinal}")

# Function to detect suspicious sections
def detect_suspicious_sections(pe):
    for section in pe.sections:
        if section.Name.decode().strip() not in [".text", ".data", ".rdata", ".rsrc"]:
            print(f"Suspicious Section: {section.Name.decode().strip()}")

# Function to detect high entropy sections
def detect_high_entropy_sections(pe):
    for section in pe.sections:
        data = section.get_data()
        if data:
            entropy = -sum((b/len(data)) * math.log2(b/len(data)) for b in data if b > 0) 
            if entropy > 7:
                print(f"High Entropy Section: {section.Name.decode().strip()}")

# Function to detect suspicious imports
def detect_suspicious_imports(pe):
    suspicious_apis = ['CreateRemoteThread', 'VirtualAlloc', 'WriteProcessMemory']
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            for imp in entry.imports:
                if imp.name and imp.name.decode() in suspicious_apis:
                    print(f"Suspicious API: {imp.name.decode()} in {entry.dll.decode()}")

# Main function to analyze a single PE file
def analyze_pe(file_path):
    try:
        pe = pefile.PE(file_path)
        print(f"Analyzing file: {file_path}")
        detect_suspicious_sections(pe)
        detect_high_entropy_sections(pe)
        detect_suspicious_imports(pe)
    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")

def automate_analysis(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".exe") or file.endswith(".dll") or file.endswith(".exe3"):
                file_path = os.path.join(root, file)
                analyze_pe(file_path)
                print("-" * 60)

if __name__ == "__main__":
    pathee = r"C:\Users\bertr\OneDrive - NEXT Uddannelse København\Skrivebord\virus-ai-detection\exe to train\AutoClicker-3.0.exe3"
    parse_pe(pathee)
    enumerate_imports(pathee)
    #automate_analysis(pathee)