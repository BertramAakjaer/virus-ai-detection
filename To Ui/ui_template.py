import model_runner as lm

if __name__ == "__main__":
    model_dir = r'C:\Users\bertr\OneDrive - NEXT Uddannelse København\Skrivebord\virus-ai-detection\AI training\[SVM] trained_models(2025-04-24 09-20-49)'
    lm.init_model(model_dir)
    
    file = r'D:\Data\[4] Harmless\Maple2024WindowsX64GYMUpgrade_da.exeh'
    is_malware, certainty, malware_certainty = lm.run_model(file)
    print(f"Is malware: {is_malware}, Certainty: {certainty}, Malware Certainty: {malware_certainty}")
    
    
    
    # Resten er bare for at vise funktionalitet
    
    brug_tærskel = False #optional evt setting
    tærskel = 0.8
    
    # Certainty er afhængig af om det er malware eller clean, derfor er malware_certainty der ogs
    
    if brug_tærskel:
        if malware_certainty > tærskel:
            print("-\t File Evaluation: Malware (med tærskel)")
        else:
            print("-\t File Evaluation: Clean (med tærskel)")
    else:
        if is_malware:
            print("-\t File Evaluation: Malware")
        else:
            print("-\t File Evaluation: Clean")