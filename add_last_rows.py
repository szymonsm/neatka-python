import os
import re
import csv
from glob import glob

base_dir = "examples/lunar-lander/parameter_sweep_results/20250520_081847"
folder_patterns = ["kan_*", "feedforward_*"]

# Regex patterns
log_filename_re = re.compile(r"log_(\d+)\.txt")
generation_re = re.compile(r"\*{6} Running generation (\d+) \*{6}")
avg_fitness_re = re.compile(r"Population's average fitness:\s+([0-9.+-eE]+)")
best_fitness_re = re.compile(r"Best fitness:\s+([0-9.+-eE]+)")
species_re = re.compile(r"species\s+(\d+)")
complexity_re = re.compile(r"size:\s+\((\d+),\s*(\d+)\)")
final_fitness_re = re.compile(r"^Fitness:\s+([0-9.+-eE]+)")
hidden_node_re = re.compile(r"\t(\d+)\sKANNodeGene")

for pattern in folder_patterns:
    for subfolder in glob(os.path.join(base_dir, pattern)):
        log_files = glob(os.path.join(subfolder, "log_*.txt"))

        for log_path in log_files:
            log_name = os.path.basename(log_path)
            match = log_filename_re.match(log_name)
            if not match:
                continue

            number = match.group(1)
            csv_path = os.path.join(subfolder, f"results_{number}.csv")
            if not os.path.exists(csv_path):
                continue  # skip extinct

            # Read log file
            with open(log_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Extract values from log
            generation = best_fitness = avg_fitness = species = None
            complexity = (None, None)
            final_fitness = None
            hidden_nodes_count = 0

            for line in reversed(lines):
                if generation is None:
                    m = generation_re.search(line)
                    if m:
                        generation = int(m.group(1))

                if avg_fitness is None:
                    m = avg_fitness_re.search(line)
                    if m:
                        avg_fitness = float(m.group(1))

                if best_fitness is None:
                    m = best_fitness_re.search(line)
                    if m:
                        best_fitness = float(m.group(1))

                if species is None:
                    m = species_re.search(line)
                    if m:
                        species = int(m.group(1))

                if complexity == (None, None):
                    m = complexity_re.search(line)
                    if m:
                        complexity = (int(m.group(1)), int(m.group(2)))

                if final_fitness is None:
                    m = final_fitness_re.match(line.strip())
                    if m:
                        final_fitness = float(m.group(1))

                m = hidden_node_re.match(line)
                if m:
                    node_key = int(m.group(1))
                    if node_key >= 300:
                        hidden_nodes_count += 1

            # Ensure all needed data is present
            if None in (generation, avg_fitness, best_fitness, species, final_fitness) or complexity == (None, None):
                print(f"Skipping incomplete log: {log_path}")
                continue

            # Compose row (with multiplied complexity)
            complexity_product = complexity[0] * complexity[1]

            new_row = [
                generation,
                f"{final_fitness:.5f}",
                f"{avg_fitness:.5f}",
                species,
                complexity_product,
                hidden_nodes_count
            ]

            # Read header
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader)
                rows = list(reader)

            # Append row
            rows.append(new_row)

            with open(csv_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)

print("All available logs processed and CSVs updated.")
