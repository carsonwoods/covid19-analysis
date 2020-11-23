# Script to Analyze Results

driving_average = 0
walking_average = 0
residential_average = 0
workplace_average = 0
apple_count = 0
google_count = 0

for root, subdirs, files in os.walk('./results/'):
    for file in files:
        if file.endswith("regression_performance.txt"):
            with open(os.path.join(root, file), "r") as fp:
                line = fp.readline()
                while line:
                    line = line.split()
                    if 'Driving' in line:
                        apple_count += 1
                        driving_average += float(line[-1])
                    elif 'Walking' in line:
                        walking_average += float(line[-1])
                    elif 'Residential' in line:
                        google_count += 1
                        residential_average += float(line[-1])
                    elif 'Workplace' in line:
                        workplace_average += float(line[-1])
                    line = fp.readline()

        if file.endswith('model_performance.txt'):
            with open(os.path.join(root, file), "r") as fp:
                line = fp.readline()
                while line:
                    line = line.split()
                    if 'Driving' in line:
                        apple_count += 1
                        driving_average += float(line[-1])
                    elif 'Walking' in line:
                        walking_average += float(line[-1])
                    elif 'Residential' in line:
                        google_count += 1
                        residential_average += float(line[-1])
                    elif 'Workplace' in line:
                        workplace_average += float(line[-1])
                    line = fp.readline()

print("\nDriving Average: " + str(driving_average/apple_count))
print("Walking Average: " + str(walking_average/apple_count))
print("Residential Average: " + str(residential_average/google_count))
print("Workplace Average: " + str(workplace_average/google_count))