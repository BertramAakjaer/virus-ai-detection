import sys
from tqdm import tqdm

class TqdmOnly:
    def __init__(self):
        self.tqdm_output = sys.__stdout__
        self.print_count = 0

    def write(self, message):
        # Count print statements (excluding tqdm-related messages)
        if not ("tqdm" in message or message.startswith('\r') or message.startswith('\n')):
            self.print_count += 1
            return
        # Only allow tqdm-related messages
        self.tqdm_output.write(message)
        self.tqdm_output.flush()

    def flush(self):
        self.tqdm_output.flush()

    def get_print_count(self):
        return self.print_count

# Create an instance of TqdmOnly
tqdm_only = TqdmOnly()

# Redirect stdout to the TqdmOnly instance
sys.stdout = tqdm_only

# Example usage with tqdm
for i in tqdm(range(10), desc="Processing"):
    # Simulate some work
    pass

# Print statements that will be counted but not shown
print("This will not be printed.")
print("This also will not be printed.")

# Get and display the count of intercepted print statements
original_stdout = sys.__stdout__
print(f"Number of print statements intercepted: {tqdm_only.get_print_count()}", file=original_stdout)