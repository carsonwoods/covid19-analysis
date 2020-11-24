# Script to Analyze Results

import os

driving_average = 0
walking_average = 0
residential_average = 0
workplace_average = 0
apple_count = 0
google_count = 0

rnn_val_loss_average = 0
rnn_perf_loss_average = 0
rnn_val_mean_sq_loss = 0
rnn_perf_mean_sq_loss = 0

lstm_val_loss_average = 0
lstm_perf_loss_average = 0
lstm_val_mean_sq_loss = 0
lstm_perf_mean_sq_loss = 0

gru_val_loss_average = 0
gru_perf_loss_average = 0
gru_val_mean_sq_loss = 0
gru_perf_mean_sq_loss = 0


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
            ml_count += 1
            with open(os.path.join(root, file), "r") as fp:
                line = fp.readline()
                while line:
                    line = line.split()
                    model_type = None
                    if 'RNN_MODEL:' in line:
                        model_type = 'rnn'
                    elif 'LSTM_MODEL:' in line:
                        model_type = 'lstm'
                    elif 'GRN_MODEL:' in line:
                        model_type = 'gru'
                    elif 'Val' in line:
                        ml_count += 1
                        if model_type = 'rnn':
                            rnn_val_loss_average += float(line[1].strip("[], "))
                            rnn_val_mean_sq_loss += float(line[2].strip("[], "))
                        elif model_type = 'lstm':
                            lstm_val_loss_average += float(line[1].strip("[], "))
                            lstm_val_mean_sq_loss += float(line[2].strip("[], "))
                        elif model_type = 'gru':
                            gru_val_loss_average += float(line[1].strip("[], "))
                            gru_val_mean_sq_loss += float(line[2].strip("[], "))
                    elif 'Val' not in line and 'Performance' in line:
                        ml_count += 1
                        perf_loss_average += float(line[1].strip("[], "))
                        perf_mean_sq_loss += float(line[2].strip("[], "))

                    line = fp.readline()

print("REGRESSION AVERAGES:")
print("Driving Average: " + str(driving_average/apple_count))
print("Walking Average: " + str(walking_average/apple_count))
print("Residential Average: " + str(residential_average/google_count))
print("Workplace Average: " + str(workplace_average/google_count))

print("\n\n")
print("ML AVERAGES:")
print("Validation Loss Average:" + str(val_loss_average/ml_count))
print("Performance Loss Average: " + str(perf_loss_average/ml_count))