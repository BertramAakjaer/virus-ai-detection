import os

def find_exe_files(folder_path):
    exe_files = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.exe', '.exev', '.exeh', '.exet')):
                exe_files.append(os.path.join(root, file))
    return exe_files

def full_scan():
    executables = []
    
    paths_to_scan = [
        os.path.join(os.path.expanduser("~"), "Downloads"),
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.join(os.environ.get("TEMP")),
        os.path.join(os.environ.get("APPDATA"), "Microsoft", "Windows", "Start Menu", "Programs", "Startup"),
        os.path.join(os.path.expanduser("~"), "AppData", "Recent"),
        os.path.join(os.environ.get("PROGRAMDATA"), "Microsoft", "Windows", "Start Menu", "Programs"),
        os.path.join("C:\\", "Windows", "System32"),
        os.path.join("C:\\", "Windows", "SysWOW64"),
        os.path.join(os.environ.get("LOCALAPPDATA"), "Temp"),
        os.path.join(os.environ.get("PROGRAMDATA")),
        os.path.join(os.environ.get("APPDATA"), "Local"),
        os.path.join(os.environ.get("APPDATA"), "Roaming") 
    ]
    
    for path in paths_to_scan:
        if os.path.exists(path):
            executables.extend(find_exe_files(path))
    
    return executables

def quick_scan():
    executables = []
    
    paths_to_scan = [
        os.path.join(os.path.expanduser("~"), "Downloads"),
        os.path.join(os.path.expanduser("~"), "Desktop"),
        os.path.join(os.environ.get("TEMP")),
        os.path.join(os.environ.get("APPDATA"), "Microsoft", "Windows", "Start Menu", "Programs", "Startup"),
        os.path.join(os.path.expanduser("~"), "AppData", "Recent"),
        os.path.join(os.environ.get("PROGRAMDATA"), "Microsoft", "Windows", "Start Menu", "Programs"),
        os.path.join(os.environ.get("LOCALAPPDATA"), "Temp"),
        os.path.join(os.environ.get("APPDATA"), "Local"),
        os.path.join(os.environ.get("APPDATA"), "Roaming") 
    ]
    
    for path in paths_to_scan:
        if os.path.exists(path):
            executables.extend(find_exe_files(path))
    
    return executables


if __name__ == "__main__":
    # custom folder path scan
    #folder_path = r"C:\Users\bertr\Downloads"
    #exe_files = find_exe_files(folder_path)

    # quick scan
    exe_files = quick_scan()

    # full scan
    #exe_files = full_scan()



    if exe_files:
        print("Found executable files:")
        for file in exe_files:
            print(file)
    else:
        print("No executable files found.")