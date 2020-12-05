# Script to Analyze Results

import os

driving_average = 0
walking_average = 0
residential_average = 0
workplace_average = 0
apple_count = 0
google_count = 0

driving_p = 0
walking_p = 0
residential_p = 0
workplace_p = 0

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

ml_count = 0

results = dict()

for root, subdirs, files in os.walk('./results/'):
    for file in files:
        if file.endswith("regression_performance.txt"):
            country = file.split("_")[0].strip()
            results[country] = dict()
            with open(os.path.join(root, file), "r") as fp:
                line = fp.readline()
                while line:
                    unsplit_line = line
                    line = line.split()
                    if 'Driving Regression Performance:' in unsplit_line:
                        apple_count += 1
                        driving_average += float(line[-1])
                        results[country]['driving_r_sq'] = float(line[-1])
                    elif 'Walking Regression Performance:' in unsplit_line:
                        walking_average += float(line[-1])
                        results[country]['walking_r_sq'] = float(line[-1])
                    elif 'Residential Regression Performance:' in unsplit_line:
                        google_count += 1
                        residential_average += float(line[-1])
                        results[country]['residential_r_sq'] = float(line[-1])
                    elif 'Workplace Regression Performance:' in unsplit_line:
                        workplace_average += float(line[-1])
                        results[country]['workplace_r_sq'] = float(line[-1])
                    elif 'Driving Regression Summary' in unsplit_line:
                        for x in range(15):
                            line = fp.readline()
                        driving_p += float(line.split()[4].strip())
                        results[country]['driving_p'] = float(line.split()[4].strip())
                    elif 'Walking Regression Summary' in unsplit_line:
                        for x in range(15):
                            line = fp.readline()
                        walking_p += float(line.split()[4].strip())
                        results[country]['walking_p'] = float(line.split()[4].strip())
                    elif 'Residential Regression Summary' in unsplit_line:
                        for x in range(15):
                            line = fp.readline()
                        residential_p += float(line.split()[4].strip())
                        results[country]['residential_p'] = float(line.split()[4].strip())
                    elif 'Workplace Regression Summary' in unsplit_line:
                        for x in range(15):
                            line = fp.readline()
                        workplace_p += float(line.split()[4].strip())
                        results[country]['workplace_p'] = float(line.split()[4].strip())
                    line = fp.readline()

        if file.endswith('model_performance.txt'):
            ml_count += 1
            model_type = None
            with open(os.path.join(root, file), "r") as fp:
                line = fp.readline()
                while line:
                    line = line.split()
                    if 'RNN_MODEL:' in line:
                        model_type = 'rnn'
                    elif 'LSTM_MODEL:' in line:
                        model_type = 'lstm'
                    elif 'GRU_MODEL:' in line:
                        model_type = 'gru'
                    elif 'Val' in line:
                        if model_type == 'rnn':
                            rnn_val_loss_average += float(line[2].strip("[], "))
                            rnn_val_mean_sq_loss += float(line[3].strip("[], "))
                        elif model_type == 'lstm':
                            lstm_val_loss_average += float(line[2].strip("[], "))
                            lstm_val_mean_sq_loss += float(line[3].strip("[], "))
                        elif model_type == 'gru':
                            gru_val_loss_average += float(line[2].strip("[], "))
                            gru_val_mean_sq_loss += float(line[3].strip("[], "))
                    elif 'Val' not in line and 'Performance:' in line:
                        if model_type == 'rnn':
                            rnn_perf_loss_average += float(line[1].strip("[], "))
                            rnn_perf_mean_sq_loss += float(line[2].strip("[], "))
                        elif model_type == 'lstm':
                            lstm_perf_loss_average += float(line[1].strip("[], "))
                            lstm_perf_mean_sq_loss += float(line[2].strip("[], "))
                        elif model_type == 'gru':
                            gru_perf_loss_average += float(line[1].strip("[], "))
                            gru_perf_mean_sq_loss += float(line[2].strip("[], "))
                    line = fp.readline()

print("REGRESSION AVERAGES:")
print("Driving Average: " + str(driving_average/apple_count))
print("Walking Average: " + str(walking_average/apple_count))
print("Residential Average: " + str(residential_average/google_count))
print("Workplace Average: " + str(workplace_average/google_count))

print("\nP-VALUE AVERAGES:")
print("Driving Average: " + str(driving_p/apple_count))
print("Walking Average: " + str(walking_p/apple_count))
print("Residential Average: " + str(residential_p/google_count))
print("Workplace Average: " + str(workplace_p/google_count))

print("\nML AVERAGES:")
print("RNN Validation Loss Average:" + str(rnn_val_loss_average/ml_count))
print("RNN Test Loss Average: " + str(rnn_perf_loss_average/ml_count))

print("LSTM Validation Loss Average: " + str(lstm_val_loss_average/ml_count))
print("LSTM Test Loss Average: " + str(lstm_perf_loss_average/ml_count))

print("GRU Validation Loss Average: " + str(gru_val_loss_average/ml_count))
print("GRU Test Loss Average: " + str(gru_perf_loss_average/ml_count))

for country in results.keys():
    if results[country]['driving_p'] >= .05:
        del results[country]['driving_p']
        del results[country]['driving_r_sq']

    if results[country]['walking_p'] >= .05:
        del results[country]['walking_p']
        del results[country]['walking_r_sq']

    try:
        if results[country]['residential_p'] >= .05:
            del results[country]['residential_p']
            del results[country]['residential_r_sq']

        if results[country]['workplace_p'] >= .05:
            del results[country]['workplace_p']
            del results[country]['workplace_r_sq']
    except:
        print("Google Data Not Included")

print(results)