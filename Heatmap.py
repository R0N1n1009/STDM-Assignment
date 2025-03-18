import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Read the data
df = pd.read_csv("D:/ucl/Spatial-Temporal Data Analysis and Data Mining/STDMAssignment/STDM-Assignment/Data5.csv",
                 parse_dates=['transit_timestamp'],
                 date_format='%m/%d/%Y %I:%M:%S %p')

# 2. Set the time index
df.set_index('transit_timestamp', inplace=True)

# 3. Group by date and station to summarize ridership
df['date'] = df.index.date
df_grouped = df.groupby(['date', 'station_complex'])['ridership'].sum().reset_index()

# 4. Create a pivot table
pivot_table = df_grouped.pivot(index='station_complex', columns='date', values='ridership')
pivot_table = pivot_table.fillna(0)  # Fill missing values with 0

# 5. Replace index with numbers (remove station names)
pivot_table_reset = pivot_table.reset_index()
pivot_table_reset['station'] = range(1, len(pivot_table_reset) + 1)  # Create numeric index
pivot_table_reset = pivot_table_reset.drop(columns=['station_complex'])  # Drop station name column
pivot_table_final = pivot_table_reset.set_index('station')  # Set numeric index as index

# 6. Plot the heatmap with a green gradient
plt.figure(figsize=(15, 12))
sns.heatmap(pivot_table_final, cmap='Greens',  # Use green gradient
            vmin=2000, vmax=8000,
            cbar_kws={'label': 'Flow'})

# 7. Set title and labels
plt.title('Flow Heatmap', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Station', fontsize=12)
plt.xticks(rotation=45)  # Rotate x-axis labels
plt.tight_layout()  # Adjust layout automatically

# 8. Display the plot
plt.show()