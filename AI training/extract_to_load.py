import math

def get_SizeOfCode(pe):
    if hasattr(pe, 'OPTIONAL_HEADER'):
        return pe.OPTIONAL_HEADER.SizeOfCode
    print("No SizeOfCode found")
    return None

def get_SizeOfInitializedData(pe):
    if hasattr(pe, 'OPTIONAL_HEADER'):
        return pe.OPTIONAL_HEADER.SizeOfInitializedData
    print("No SizeOfInitializedData found")
    return None

def get_SizeOfImage(pe):
    if hasattr(pe, 'OPTIONAL_HEADER'):
        return pe.OPTIONAL_HEADER.SizeOfImage
    print("No SizeOfImage found")
    return None

def get_Subsystem(pe):
    if hasattr(pe, 'OPTIONAL_HEADER'):
        return pe.OPTIONAL_HEADER.Subsystem
    print("No Subsystem found")
    return None

def get_EntropyCalculation_and_sections(pe):
    entropy_dict = {}

    for section in pe.sections:
        data = section.get_data()

        if not data:
            continue

        byte_counts = [0] * 256

        for byte in data:
            byte_counts[byte] += 1

        entropy = 0

        for count in byte_counts:
            if count:
                p_x = count / len(data)
                entropy -= p_x * math.log2(p_x)

        entropy_dict[section.Name.decode().strip()] = entropy
    
    if len(entropy_dict) == 0:
        print("No entropy found")
        return None
        
    return entropy_dict

def get_Imported_DLLs(pe):
    imported_dlls = {}
    if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
        for entry in pe.DIRECTORY_ENTRY_IMPORT:
            dll_name = entry.dll.decode()
            imported_dlls[dll_name] = []
            for imp in entry.imports:
                if imp.name:
                    imported_dlls[dll_name].append(imp.name.decode())
    
    if len(imported_dlls) == 0:
        print("No imported DLLs found")
        return None
        
    return imported_dlls
