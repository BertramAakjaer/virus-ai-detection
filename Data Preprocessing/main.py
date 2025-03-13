import os
import json
import pefile

def extract_and_save_metadata(folder_path):
    # Iterate over all files in the given folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.exe3'):
            file_path = os.path.join(folder_path, filename)
            txt_file_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}.txt")
            try:
                # Parse the EXE file
                pe = pefile.PE(file_path)

                # Extract metadata
                metadata = {
                    'Filename': filename,
                    'Machine': pe.FILE_HEADER.Machine,
                    'TimeDateStamp': pe.FILE_HEADER.TimeDateStamp,
                    'Characteristics': pe.FILE_HEADER.Characteristics,
                    'Subsystem': pe.OPTIONAL_HEADER.Subsystem,
                    'SizeOfImage': pe.OPTIONAL_HEADER.SizeOfImage,
                    'EntryPoint': pe.OPTIONAL_HEADER.AddressOfEntryPoint,
                    'Sections': [
                        {
                            'Name': section.Name.decode('utf-8', errors='ignore'),
                            'SizeOfRawData': section.SizeOfRawData,
                            'Characteristics': section.Characteristics
                        }
                        for section in pe.sections
                    ],
                    'Imports': [],
                    'Exports': [],
                    'Resources': [],
                    'Debug Info': None,
                    'TLS Callbacks': None,
                    'Load Configuration': None,
                    'Bound Imports': None,
                }

                # Extract imports
                if hasattr(pe, 'DIRECTORY_ENTRY_IMPORT'):
                    for entry in pe.DIRECTORY_ENTRY_IMPORT:
                        for imp in entry.imports:
                            metadata['Imports'].append({
                                'DLL': entry.dll.decode('utf-8', errors='ignore'),
                                'Function': imp.name.decode('utf-8', errors='ignore') if imp.name else 'Ordinal',
                                'Address': hex(imp.address)
                            })

                # Extract exports
                if hasattr(pe, 'DIRECTORY_ENTRY_EXPORT'):
                    for exp in pe.DIRECTORY_ENTRY_EXPORT.symbols:
                        metadata['Exports'].append({
                            'Name': exp.name.decode('utf-8', errors='ignore'),
                            'Address': hex(pe.OPTIONAL_HEADER.ImageBase + exp.address)
                        })

                # Extract resources
                if hasattr(pe, 'DIRECTORY_ENTRY_RESOURCE'):
                    for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
                        metadata['Resources'].append({
                            'Type': pefile.RESOURCE_TYPE.get(resource_type.id, 'Unknown'),
                            'Entries': len(resource_type.directory.entries)
                        })

                # Extract debug information
                if hasattr(pe, 'DIRECTORY_ENTRY_DEBUG'):
                    for dbg in pe.DIRECTORY_ENTRY_DEBUG:
                        metadata['Debug Info'] = {
                            'Type': dbg.struct.Type,
                            'Characteristics': dbg.struct.Characteristics,
                            'AddressOfRawData': dbg.struct.AddressOfRawData,
                            'SizeOfData': dbg.struct.SizeOfData
                        }

                # Extract TLS callbacks
                if hasattr(pe, 'DIRECTORY_ENTRY_TLS'):
                    metadata['TLS Callbacks'] = {
                        'Callbacks': [hex(callback) for callback in pe.DIRECTORY_ENTRY_TLS.struct.AddressOfCallBacks]
                    }

                # Extract load configuration
                if hasattr(pe, 'DIRECTORY_ENTRY_LOAD_CONFIG'):
                    metadata['Load Configuration'] = {
                        'Characteristics': pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Characteristics,
                        'Size': pe.DIRECTORY_ENTRY_LOAD_CONFIG.struct.Size
                    }

                # Extract bound imports
                if hasattr(pe, 'DIRECTORY_ENTRY_BOUND_IMPORT'):
                    metadata['Bound Imports'] = [
                        bound_imp.dll.decode('utf-8', errors='ignore')
                        for bound_imp in pe.DIRECTORY_ENTRY_BOUND_IMPORT
                    ]

                # Save metadata to a text file in JSON format
                with open(txt_file_path, 'w') as txt_file:
                    json.dump(metadata, txt_file, indent=4)

            except Exception as e:
                print(f"Error processing {filename}: {e}")

folder_path = r'C:\Users\bertr\OneDrive - NEXT Uddannelse København\Skrivebord\virus-ai-detection\exe to train'
extract_and_save_metadata(folder_path)
