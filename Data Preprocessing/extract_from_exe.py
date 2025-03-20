import pefile, math

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

def get_section_names(pe):
    section_names = []
    for section in pe.sections:
        section_names.append(section.Name.decode().strip())
    
    if len(section_names) == 0:
        print("No sections found")
        return None
        
    return section_names

def get_EntropyCalculation(pe):
    entropy_list = []
    for section in pe.sections:
        data = section.get_data()
        if data:
            entropy = -sum((b/len(data)) * math.log2(b/len(data)) for b in data if b > 0) 
            entropy_list.append(entropy)
    
    if len(entropy_list) == 0:
        print("No entropy found")
        return None
        
    return entropy_list

def get_Importet_DLLs(pe):
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