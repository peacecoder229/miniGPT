#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import configparser

def extract_int_from_str_col(col):
    """Extract integer from string column."""
    return col.str.extract('(\d+)').fillna(0).astype(int)

#def plot_charts(args, files, metric_to_plot, group_by, chart_type):
def plot_charts(args, files, metric_to_plot, sec_plot, group_by, chart_type, sec_plot_type):

    config = configparser.ConfigParser()
    if args.config:
        config.read(args.config)

    title = config.get('PlotParameters', 'title', fallback="Default Title")
    primary_axis_title = config.get('PlotParameters', 'primary_axis_title', fallback="Default Primary Axis Title")
    secondary_axis_title = config.get('PlotParameters', 'secondary_axis_title', fallback="Default Secondary Axis Title")
    xaxis_title = config.get('PlotParameters', 'xaxis_title', fallback="Default X Axis Title")
    primary_log_scale = config.getboolean('PlotParameters', 'primary_log_scale', fallback=False)
    secondary_log_scale = config.getboolean('PlotParameters', 'secondary_log_scale', fallback=False)


    title_font_size = config.getint('FontSizes', 'title', fallback=28)
    primary_axis_font_size = config.getint('FontSizes', 'primary_axis_title', fallback=24)
    secondary_axis_font_size = config.getint('FontSizes', 'secondary_axis_title', fallback=24)
    xaxis_font_size = config.getint('FontSizes', 'xaxis_title',  fallback=24)

    title_style = config.get('FontStyles', 'title', fallback="normal")
    primary_axis_style = config.get('FontStyles', 'primary_axis_title', fallback="normal")
    secondary_axis_style = config.get('FontStyles', 'secondary_axis_title', fallback="normal")
    xaxis_style = config.get('FontStyles', 'xaxis_title', fallback="normal")


    # For legend parameters
    primary_legend_location = config.get('LegendParameters', 'primary_legend_location', fallback="center")
    primary_legend_bbox_x = float(config.get('LegendParameters', 'primary_legend_bbox_x', fallback="0.5"))
    primary_legend_bbox_y = float(config.get('LegendParameters', 'primary_legend_bbox_y', fallback="0.75"))
    primary_legend_fontsize = int(config.get('FontSizes', 'primary_legend_fontsize', fallback=20))

    secondary_legend_location = config.get('LegendParameters', 'secondary_legend_location', fallback="upper left")
    secondary_legend_fontsize = int(config.get('FontSizes', 'secondary_legend_fontsize', fallback=24))
    secondary_legend_bbox_x = float(config.get('LegendParameters', 'secondary_legend_bbox_x', fallback="0.1"))
    secondary_legend_bbox_y = float(config.get('LegendParameters', 'secondary_legend_bbox_y', fallback="1.0"))

    # For tick parameters
    primary_ytick_labelsize = int(config.get('TickParameters', 'primary_ytick_labelsize', fallback=15))
    secondary_ytick_labelsize = int(config.get('TickParameters', 'secondary_ytick_labelsize', fallback=20))

    show_primary_axis_label = config.getboolean('PlotParameters', 'show_primary_axis_label', fallback=False)
    show_secondary_axis_label = config.getboolean('PlotParameters', 'show_secondary_axis_label', fallback=False)
    label_fontsize = config.getint('PlotParameters', 'label_fontsize', fallback=10)
    txtbox = config.getboolean('ExtraTextBox', 'showtextbox', fallback=False)
    txtbox_x = float(config.get('ExtraTextBox', 'x_position', fallback="0.9"))
    txtbox_y = float(config.get('ExtraTextBox', 'y_position', fallback="0.9"))
    txtbox_font = config.getint('ExtraTextBox', 'font', fallback=12)
    txtbox_txt = config.get('ExtraTextBox', 'text', fallback="Default")
    # Read the files into dataframes using regex for space or tab separation
    #dataframes = [pd.read_csv(f, sep=r'\s+') for f in files]
    dataframes = [df.mask(df.applymap(lambda x: x < 0 if np.issubdtype(type(x), np.number) else False)) for df in (pd.read_csv(f, sep=r'\s+') for f in files)]

    #dataframes = [(pd.read_csv(f, sep=r'\s+')).mask(lambda x: x < 0) for f in files]

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
        if args.applynumericsort:
            for col in group_by:
                if df[col].dtype == 'object':  # only process string columns
                    df_to_sort[col] = extract_int_from_str_col(df[col])
        else:
            print("Numaric Sort is not needed for cokumn names\n")
        sorted_df = df_to_sort.sort_values(by=group_by)
        # Calculate position based on hierarchical grouping
        positions = np.arange(len(sorted_df))

        # Plot bars
        if chart_type == "bar":
            bars = ax.bar(positions + file_idx * bar_width, sorted_df[metric_to_plot],
                          width=bar_width, label=f,
                          #color=['black', 'green', 'blue'][file_idx % 3],
                          #color=['black', 'yellow', 'blue'][file_idx % 3],
                          color=['lightblue', 'yellow', 'blue', 'green', 'black'][file_idx % 5],
                          edgecolor=['black', 'black', 'blue'][file_idx % 3],
                          linestyle=['-', '--', '-.'][file_idx % 3], alpha=0.7)


            y_min = sorted_df[metric_to_plot].min(skipna=True)  # skipping NaN values
            y_max = sorted_df[metric_to_plot].max(skipna=True)
            y_mid = (y_min + y_max) / 100


            for pos, value in zip(positions, sorted_df[metric_to_plot]):
                if pd.isna(value):  # check if value is nan
                    marker_size = bar_width * 5000  # adjust the size to fit the text inside
                    x = pos + file_idx * bar_width
                    y = y_mid  # adjust the vertical position as needed

                    # Create and add the hexagon to the plot using scatter plot
                   # ax.scatter(x, y, marker='H', s=marker_size, facecolors='none', edgecolors='yellow', linewidths=2, alpha=0.7)
                    ax.scatter(x, y, marker=(6, 0, 0), s=marker_size, facecolors='none', edgecolors='black', linewidths=2, alpha=0.7)
                    # Add OOM text at the center of the hexagon
                    ax.text(x, y, 'OOM', ha='center', va='center', fontsize=14, color='red')


            if show_primary_axis_label:
                for rect in bars:
                    height = rect.get_height()
                    if not np.isnan(height):
                        ax.text(rect.get_x() + rect.get_width()/2., height, '%d' % int(height),
                            ha='center', va='bottom', fontsize=label_fontsize)

        
        if sec_plot:  # Only plot if sec_plot column is provided

            if sec_plot_type == "line":
                ax2.plot(positions, sorted_df[sec_plot], label=f, color=['black', 'green', 'blue'][file_idx % 3],
                         linestyle=['-.', '--', '-.'][file_idx % 3], marker='o')
                nan_positions = sorted_df[sorted_df[sec_plot].isna()].index
                ax2.scatter(positions[nan_positions], [0]*len(nan_positions), color='red', marker='x', label='Incomplete Data', zorder=5)
                for pos in nan_positions:
                    ax2.annotate('NaN', (positions[pos], 0), textcoords="offset points", xytext=(0,10), ha='center')
            elif sec_plot_type == "symbol":  # New condition for the new plot type
                #marker_size = 50
                draw = 'Circle'
                #draw = 'squre'
                edge = 'green'
                txtcolor = 'green'
                symbol_shape = 'o' if draw == 'Circle' else 's'
                y_min = sorted_df[sec_plot].min(skipna=True)  # skipping NaN values
                y_max = sorted_df[sec_plot].max(skipna=True)
                y_mid = (y_min + y_max) / 1.5
                # you can change this value to adjust the size of the marker
                for x, y in zip(positions, sorted_df[sec_plot]):
                    if not np.isnan(y):  # Display only if the value is not NaN
                        #ax2.scatter(x, y, marker=symbol_shape, s=marker_size, color='purple', label=f, alpha=0.7)
                        #ax2.text(x, y, '%d' % int(y), ha='center', va='center', fontsize=label_fontsize)
                        text_str = '%d' % int(y)
                        marker_size = max(len(text_str) * 200, 150)
                        text_fontsize = 11
                    else:
                        text_str = 'OOM'
                        symbol_shape = 'H'
                        edge = 'yellow' 
                        txtcolor = 'red'
                        y = y_mid  # Replace with the midpoint
                        marker_size = max(len(text_str) * 300, 100)
                        text_fontsize = 14  # you can adjust this to change the text size
                    #marker_size = len(text_str) * 100  # 100 is a multiplier to adjust the size proportionally to the text length
                    ax2.scatter(x, y, marker=symbol_shape, s=marker_size, facecolors='none', edgecolors=edge, linewidths=2, label=f, alpha=0.7)
                    ax2.text(x, y, text_str, ha='center', va='center', fontsize=text_fontsize, color=txtcolor)

            else:  # Default is bar
                ax2.bar(positions + file_idx * bar_width, sorted_df[sec_plot], width=bar_width,
                        color=['red', 'green', 'blue'][file_idx % 3], label=f, alpha=0.7)

            if show_secondary_axis_label:

                if sec_plot_type == "line":
                    for x, y in zip(positions, sorted_df[sec_plot]):
                        if not np.isnan(y):
                            ax2.text(x, y, '%d' % int(y), ha='center', va='bottom', fontsize=label_fontsize)
                elif sec_plot_type == "symbol":
                    pass

                else:
                    bars_sec = ax2.bar(positions + file_idx * bar_width, sorted_df[sec_plot], width=bar_width,
                                       color=['red', 'green', 'blue'][file_idx % 3], alpha=0.7)
                    for rect in bars_sec:
                        height = rect.get_height()
                        ax2.text(rect.get_x() + rect.get_width()/2., height, '%d' % int(height),
                                 ha='center', va='bottom', fontsize=label_fontsize)



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






# For the title:
    ax.set_title(title, fontsize=title_font_size, fontweight=title_style)

# For the primary axis:
    ax.set_ylabel(primary_axis_title, fontdict={'size': primary_axis_font_size, 'weight': primary_axis_style})



    #ax.set_xticklabels(dataframes[0][group_by].astype(str).agg(' '.join, axis=1).tolist(), rotation=45, ha='right')
    plt.subplots_adjust(bottom=0.35)
    #ax.set_xlabel('GPTparam(inMillion)_BS_MaxTokens_instanceCount', fontdict=font_properties)
    #ax.set_xlabel('instanceCount_GeneratedTokens', fontdict=font_properties)
    ax.set_xlabel(xaxis_title, fontsize=xaxis_font_size, fontweight=xaxis_style)
    #ax.set_ylabel('Per Token latency in (ms)', fontdict=font_properties)
    #ax.set_ylabel('Throughput : Total Tokens per Second', fontdict=font_properties)
    #ax.set_ylabel('Memory utilized (GB)', fontdict=font_properties)
    ax.tick_params(axis='y', labelsize=primary_ytick_labelsize)
    #ax.legend(loc="upper right", fontsize=20)
    prim_leg = ax.legend(loc=primary_legend_location, bbox_to_anchor=(primary_legend_bbox_x, primary_legend_bbox_y), fontsize=primary_legend_fontsize)
    if primary_log_scale:
        ax.set_yscale('log')


    #prim_leg.set_visible(False)
    if sec_plot:
        #ax2.set_ylabel("Total Tokens per second", fontsize=28)
        ax2.set_ylabel(secondary_axis_title, fontsize=secondary_axis_font_size, fontweight=secondary_axis_style)
        sec_leg = ax2.legend(loc=secondary_legend_location, bbox_to_anchor=(secondary_legend_bbox_x, secondary_legend_bbox_y), fontsize=secondary_legend_fontsize)
        sec_leg.set_visible(False)
        ax2.tick_params(axis='y', labelsize=secondary_ytick_labelsize)
        if secondary_log_scale:  # Only set log scale if there is a secondary plot
            ax2.set_yscale('log')

    #fig.text(0.4, 0.75, 'line plots -> Throughput \nbars -> Latency per Token', fontsize=20, 
    #                     verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
    if txtbox:
        fig.text(txtbox_x, txtbox_y, txtbox_txt, fontsize=txtbox_font, verticalalignment='center', bbox=dict(facecolor='white', alpha=0.7))
    plt.tight_layout()
    file_format = args.outfile.split('.')[-1]
    plt.savefig(args.outfile, format=file_format)

    #plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot charts based on input files and parameters.')
    parser.add_argument('--files', required=True, help='Comma-separated list of files to read data from.')
    parser.add_argument('--applynumericsort', action="store_true", help='If given then only extract numaric vaue from columnname.')
    parser.add_argument('--metric-to-plot', required=True, help='Name of the metric to plot.')
    parser.add_argument('--group_by', required=True, help='Comma-separated list of columns by which to group data.')
    parser.add_argument('--chart-type', default='bar', choices=['bar', 'line'], help='Type of chart to plot (bar or line).')
    parser.add_argument('--outfile', required=True, help='Output file name to save the plot.')
    parser.add_argument('--sec_plot', default=None, help='Name of the column to plot on the secondary axis.')
    parser.add_argument('--sec_plot_type', default='bar', choices=['bar', 'line', 'symbol'], help='Type of secondary plot (bar or line).')
    parser.add_argument('--config', default=None, help='Path to the configuration file.')
    args = parser.parse_args()

    files = args.files.split(',')
    group_by = args.group_by.split(',')
    #plot_charts(args, files, args.metric_to_plot, group_by, args.chart_type)
    plot_charts(args, files, args.metric_to_plot, args.sec_plot, group_by, args.chart_type, args.sec_plot_type)

if __name__ == "__main__":
    main()

