import sys
import os
import os.path
import io
import pandas
import argparse
import math
import numpy

pandas.options.mode.chained_assignment = 'raise'

argument_parser = argparse.ArgumentParser(description='Convert UCI time series data')
argument_parser.add_argument('data_dir', help='The directory containing the data')
argument_parser.add_argument('out_file', help='The output file')
argument_parser.add_argument('--num_variables', type=int)
argument_parser.add_argument('--num_classes', type=int)
argument_parser.add_argument('--window_size', type=int, default=100)
argument_parser.add_argument('--window_stride', type=int, default=100)

arguments = argument_parser.parse_args(sys.argv[1:])
arguments.data_dir
arguments.num_variables
arguments.num_classes

def encode_one_hot(one_hot_frame, index, label):
    one_hot_array = one_hot_frame.to_numpy()
    one_hot_array[index,:] = 0.0
    one_hot_array[index,label] = 1.0

def fill_frame(window_frame, value_frame, label_frame):
    window_array = window_frame.to_numpy()
    value_array = value_frame.to_numpy()
    label_array = label_frame.to_numpy()

    window_array[:,:value_frame.shape[1]] = value_array
    window_array[:,value_frame.shape[1]:] = label_array
  

  
try:
    out = open(arguments.out_file, "w")
except OSError:
    sys.exit(f"Couldn't create the data output file {arguments.out_file}")
print(f"Created the data output file {arguments.out_file}")

try:
    os.chdir(arguments.data_dir)
except FileNotFoundError:
    sys.exit(f"Couldn't find the directory {arguments.data_dir}")
except PermissionError:
    sys.exit(f"We do not have permission to access the directory {arguments.data_dir}")
except NotADirectoryError:
    sys.exit(f"The specified directory is not a {arguments.data_dir}")

print(f"Converting data in {arguments.data_dir}")

sub_dirs = list(filter(lambda item: os.path.isdir(item), os.listdir()))

num_sub_dirs = len(sub_dirs)

print(f"Found {num_sub_dirs} class directories")
#if (arguments.num_classes != ((num_sub_dirs*2)-1)):
#  sys.exit(f"The specified number of classes doesn't match the number found")

out.write(f"{arguments.num_variables},{arguments.num_classes}\n")

# Dataframes used during processing
window_frame = pandas.DataFrame(index=range(arguments.window_size), columns=range(arguments.num_variables+arguments.num_classes))
one_hot_label_frame = pandas.DataFrame(index=range(arguments.window_size), columns=range(arguments.num_classes))


for sub_dir in sub_dirs:

    print(f"Fetching data in {sub_dir}")

    instance_files = os.listdir(sub_dir)
    sub_dir_label = int(sub_dir)

    for instance_file in instance_files:

        print(f"Fetching data for instance {instance_file}")

        instance_file = os.path.join(sub_dir, instance_file)
        instance_frame = pandas.read_csv(instance_file)
    
        instance_value_frame = instance_frame.iloc[:,1:-1]
        instance_label_series = instance_frame.iloc[:,-1]

        instance_value_frame.fillna(value=0.0, inplace=True)
        instance_label_series.fillna(method='ffill', axis=0, inplace=True)

        num_rows, num_columns = instance_frame.shape
        window_start = 0
        start_label = int(instance_label_series.iloc[0])
        start_label_ = start_label*2 if start_label<100 else (start_label-100)*2-1

        if start_label == 0 and sub_dir_label != 0:
            while int(instance_label_series.iloc[window_start+arguments.window_size]) == start_label:
                window_start = window_start+arguments.window_stride

        window_end = window_start+arguments.window_size
        end_label = int(instance_label_series.iloc[window_end])
        end_label_ = end_label*2 if end_label<100 else (end_label-100)*2-1

        value_frame = instance_value_frame.iloc[window_start:window_end,:]
 
        for index in range(arguments.window_size):
            encode_one_hot(one_hot_label_frame, index, end_label_)

        fill_frame(window_frame, value_frame, one_hot_label_frame)

        for index, row in window_frame.iterrows():
            for x in row.values[0:-1]:
                out.write(str(x)+',')
            out.write(str(row.values[-1]))
            if index<arguments.window_size-1:
                out.write(',')

        out.write('\n')

        # Continue to next instance if either start_label is something else than 0, or if sub_dir_label is equal to 0
        if start_label != 0 or sub_dir_label == 0:
            continue

        # Fetch second window

        start_label = end_label 
        start_label_ = end_label_

        while window_start+arguments.window_size<num_rows and int(instance_label_series.iloc[window_start+arguments.window_size]) == start_label:
            window_start = window_start+arguments.window_stride

        if not window_start+arguments.window_size<num_rows:
            continue

        window_end = window_start+arguments.window_size
        end_label = int(instance_label_series.iloc[window_end])
        end_label_ = end_label*2 if end_label<100 else (end_label-100)*2-1

        value_frame = instance_value_frame.iloc[window_start:window_end,:]

        for index in range(arguments.window_size):
            encode_one_hot(one_hot_label_frame, index, end_label_)

        fill_frame(window_frame, value_frame, one_hot_label_frame)

        for index, row in window_frame.iterrows():
            for x in row.values[0:-1]:
                out.write(str(x)+',')
            out.write(str(row.values[-1]))
        if index<arguments.window_size-1:
            out.write(',')

        out.write('\n')

   
    out.close()

print("Finished converting")
