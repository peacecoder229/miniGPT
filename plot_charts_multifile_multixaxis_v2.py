#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def extract_int_from_str_col(col):
    """Extract integer from string column."""
    return col.str.extract('(\d+)').fillna(0).astype(int)

#def plot_charts(args, files, metric_to_plot, group_by, chart_type):
def plot_charts(args, files, metric_to_plot, sec_plot, group_by, chart_type, sec_plot_type):
    # Read the files into dataframes using regex for space or tab separation
    dataframes = [pd.read_csv(f, sep=r'\s+') for f in files]

    # Define positions for bars
    n_files = len(files)
    bar_width = 0.2

    fig, ax = plt.subplots(figsize=(20, 10))
    if sec_plot:  # Only plot if sec_plot column is provided
        ax2 = ax.twinx()  # instantiate a second y-axis sharing the same x-axis

    for file_idx, (df, f) in enumerate(zip(dataframes, files)):
        # Sort the data based on group_by columns
        #sorted_df = df.sort_values(by=group_by, key=lambda x:x.str.extract('(\d+)').astype(int))
        df_to_sort = df.copy()
        for col in group_by:
            if df[col].dtype == 'object':  # only process string columns
                df_to_sort[col] = extract_int_from_str_col(df[col])
        sorted_df = df_to_sort.sort_values(by=group_by)
        # Calculate position based on hierarchical grouping
        positions = np.arange(len(sorted_df))

        # Plot bars
        if chart_type == "bar":
            bars = ax.bar(positions + file_idx * bar_width, sorted_df[metric_to_plot],
                          width=bar_width, label=f,
                          color=['black', 'green', 'blue'][file_idx % 3],
                          #color=['black', 'yellow', 'blue'][file_idx % 3],
                          edgecolor=['black', 'black', 'blue'][file_idx % 3],
                          linestyle=['-', '--', '-.'][file_idx % 3], alpha=0.7)
        
        if sec_plot:  # Only plot if sec_plot column is provided

            if sec_plot_type == "line":
                ax2.plot(positions, sorted_df[sec_plot], label=f, color=['black', 'green', 'blue'][file_idx % 3],
                         linestyle=['-', '--', '-.'][file_idx % 3], marker='o')
            else:  # Default is bar
                ax2.bar(positions + file_idx * bar_width, sorted_df[sec_plot], width=bar_width,
                        color=['red', 'green', 'blue'][file_idx % 3], label=f, alpha=0.7)


        if file_idx == 0:
            xtick_labels = sorted_df[group_by].apply(lambda x: '_'.join(x.astype(str)), axis=1)

            # Set x-ticks and labels
            ax.set_xticks(positions + (n_files / 2 - 0.5) * bar_width)
            ax.set_xticklabels(xtick_labels, fontsize=20, rotation=90)
    #ax.set_xticks([])  # Remove x-ticks
    #ax.set_xticklabels([])  # Remove x-tick labels

    #ypos = -0.4  # starting y-position for labels
    font_sizes = [20, 15, 8, 8]  # Adjust this list based on the number of items in group_by, if needed.
    rotation_angles = [0, 90, 90, 90]  # Rotation angle for each group level
    # Assuming sorted_df corresponds to the data from the first file
    first_file_data = dataframes[0]  # Get the DataFrame of the first file
    sorted_first_file_data = first_file_data.sort_values(by=group_by).reset_index(drop=True)

    #ax.set_xticklabels(dataframes[0][group_by].astype(str).agg(' '.join, axis=1).tolist(), rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.35)
    #ax.set_title("H100 avg and max mem used with 150mil params nanoGPT model\n BatchSize=1", fontsize=28)
    ax.set_title("64Core Xeon vs H100 GPU perf on 150mil params nanoGPT model\n BatchSize=256", fontsize=28)
    font_properties = {'weight': 'bold', 'size': 24}  # adjust as necessary
    #ax.set_xlabel('GPTparam(inMillion)_BS_MaxTokens_instanceCount', fontdict=font_properties)
    ax.set_xlabel('instanceCount_GeneratedTokens', fontdict=font_properties)
    ax.set_ylabel('Per Token latency in (ms)', fontdict=font_properties)
    #ax.set_ylabel('Throughput : Total Tokens per Second', fontdict=font_properties)
    #ax.set_ylabel('Memory utilized (GB)', fontdict=font_properties)
    ax.tick_params(axis='y', labelsize=15)
    #ax.legend(loc="upper right", fontsize=20)
    ax.legend(loc='center', bbox_to_anchor=(0.5, 0.75), fontsize=20)
    if sec_plot:
        #ax2.set_ylabel("Total Tokens per second", fontsize=28)
        ax2.set_ylabel('Peak memory utilized (GB)', fontdict=font_properties)
        ax2.legend(loc="upper left", fontsize=24)
        ax2.tick_params(axis='y', labelsize=20)

    #fig.text(0.4, 0.75, 'line plots -> Throughput \nbars -> Latency per Token', fontsize=20, 
    #                     verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    plt.savefig(args.outfile, format='pdf')


    #plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot charts based on input files and parameters.')
    parser.add_argument('--files', required=True, help='Comma-separated list of files to read data from.')
    parser.add_argument('--metric-to-plot', required=True, help='Name of the metric to plot.')
    parser.add_argument('--group_by', required=True, help='Comma-separated list of columns by which to group data.')
    parser.add_argument('--chart-type', default='bar', choices=['bar', 'line'], help='Type of chart to plot (bar or line).')
    parser.add_argument('--outfile', required=True, help='Output file name to save the plot.')
    parser.add_argument('--sec_plot', default=None, help='Name of the column to plot on the secondary axis.')
    parser.add_argument('--sec_plot_type', default='bar', choices=['bar', 'line'], help='Type of secondary plot (bar or line).')
    args = parser.parse_args()

    files = args.files.split(',')
    group_by = args.group_by.split(',')
    #plot_charts(args, files, args.metric_to_plot, group_by, args.chart_type)
    plot_charts(args, files, args.metric_to_plot, args.sec_plot, group_by, args.chart_type, args.sec_plot_type)

if __name__ == "__main__":
    main()

