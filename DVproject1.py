# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:18:17 2024

@author: jneid
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import datetime
from datetime import date
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import squarify
# squarify fo the treemap 
import pandas as pd
from math import pi
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits import mplot3d
import plotly.express as px
sns.set_style("darkgrid")
import matplotlib.ticker as ticker
df = pd.read_csv(r"C:\Users\jneid\OneDrive\Desktop\Data Visualizatiomn\DataVizProject\marketing_campaign 
- marketing_campaign.csv (1).csv")
#0-Problem :
# Convert the date to datetime objects with the correct format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
df['Total_Purchases'] = df['MntFruits'] + df['MntMeatProducts'] + 
df['MntFishProducts']+df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
# Extract the year from the purchase date
df['Year'] = df['Dt_Customer'].dt.year
# Group the data by year and calculate the average purchase for each year
average_purchase_by_year = df.groupby('Year')['Total_Purchases'].mean()/2
# Create a line chart of the average purchase over time
plt.plot(average_purchase_by_year.index, average_purchase_by_year.values, marker='o', linestyle='-' , color 
="#d780ff")
# Add data points for the average of each year
plt.scatter(average_purchase_by_year.index, average_purchase_by_year.values, color='red', label='Average 
Purchase')
# Set the x-axis label and title
plt.ylabel('Average Number of Purchases')
plt.title('Average Purchase Over the Years')
# Customize the x-axis ticks and labels
years = [2012, 2013, 2014]
plt.xticks(years, years)
# Show the plot
plt.show()
# A - Dustribution de la population : done
# 1 distribution by age
# Calculate the age of each customer based on their year of birth
df['Age'] = 2015 - df['Year_Birth']
# Create a histogram of the age column using matplotlib
plt.hist(df['Age'], bins=10, weights=np.ones(len(df['Age']))/len(df['Age']), color='#ae00ff', edgecolor='white')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Customer Age Distribution')
plt.xticks(np.arange(0, 110, 10)) # Set the ticks on the x axis from 0 to 110 with an interval of 10
# Get the patches object from the histogram
patches = plt.gca().patches
# Get the max value from the histogram
max_value = max([p.get_height() for p in patches])
# Loop through the patches and change the color of the max value
for p in patches:
 if p.get_height() == max_value:
 p.set_facecolor('#f7e6ff')
plt.show()
#1.2 marital status
# Define the color palette
colors = ['#f7e6ff','#efccff', '#e7b3ff' , '#df99ff' ,'#d780ff']
# Group the smaller values together
data = df["Marital_Status"].value_counts()
labels = [x if data[x] >= 5 else "others" for x in data.index]
data = data.groupby(labels).sum()
# Set the explode values
explode = [0.1 if x < max(data) else 0 for x in data]
# Create a pie chart with matplotlib
plt.pie(data, labels=labels, explode=explode, shadow=True, autopct="%1.1f%%", colors=colors)
# Set the title and the equal aspect ratio
plt.title("Marital Status Distribution")
plt.axis("equal")
# Show the chart
plt.show()
#1.3 treemap for education level : using squarify library 
counts = df['Education'].value_counts()
labels = counts.index
sizes = counts.values
colors = ['#f7e6ff','#efccff', '#e7b3ff' , '#df99ff' ,'#d780ff']
squarify.plot(sizes=sizes, label=labels, color=colors)
plt.title('Treemap of Education')
plt.axis('off')
plt.show() 
#1-4 Income Distribution
# Create a figure with a custom size and resolution
plt.figure(figsize=(10, 6), dpi=100)
df = df[(df['Income'] <= 170000) & (df['Income'] >= 1000)]
# Create a histogram of age for the high-income population
plt.hist(df["Income"], bins=range(0, max(df["Income"]) + 1000, 10000), color="#efccff", edgecolor="white")
# Add labels and a title to the plot
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.title("Income Distribution for the Population")
tick_spacing = 10000
plt.xticks(range(0, max(df["Income"]) + 10000, tick_spacing), rotation='vertical')
# Format ticks to display in k units
plt.gca().xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:,.0f}k'.format(x / 1000)))
# Add vertical grid lines for better visibility
plt.grid(axis='x', linestyle='--', alpha=0.6)
# Show the plot
plt.show()
# 1.5 heatmap of avg spent on product by household composition 
products = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 
'MntGoldProds']
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for i, product in enumerate(products):
 row = i // 3
 col = i % 3
 
 grouped = df.groupby(['Kidhome', 'Teenhome'])[product].mean().reset_index()
 matrix = grouped.pivot(index='Kidhome', columns='Teenhome', values=product)
 sns.heatmap(matrix, annot=True, cmap='Blues', ax=axes[row, col])
 axes[row, col].set_xlabel('Number of Teenagers in Household')
 axes[row, col].set_ylabel('Number of Children in Household')
 axes[row, col].set_title(f'{product}')
plt.tight_layout()
plt.show()
# 1.6 stacked bar chart for if you want to create a stacked bar chart of the AcceptedCmp1, AcceptedCmp2, 
AcceptedCmp3,
#AcceptedCmp4, AcceptedCmp5, and Response attributes by month of enrollment,
# Convert the Dt_Customer attribute to datetime format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'])
# Extract the month of enrollment from the Dt_Customer attribute
df['Month'] = df['Dt_Customer'].dt.month
# Group the data by month and calculate the sum of accepted campaigns
grouped = df.groupby('Month')[['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 
'AcceptedCmp5', 'Response']].sum()
# Get the labels, heights, and positions for each month
labels = grouped.index
heights1 = grouped['AcceptedCmp1'].values
heights2 = grouped['AcceptedCmp2'].values
heights3 = grouped['AcceptedCmp3'].values
heights4 = grouped['AcceptedCmp4'].values
heights5 = grouped['AcceptedCmp5'].values
heights6 = grouped['Response'].values
positions = range(len(labels))
# Create a stacked bar chart using matplotlib
plt.bar(positions, heights1, label='Campaign 1')
plt.bar(positions, heights2, bottom=heights1, label='Campaign 2')
plt.bar(positions, heights3, bottom=heights1+heights2, label='Campaign 3')
plt.bar(positions, heights4, bottom=heights1+heights2+heights3, label='Campaign 4')
plt.bar(positions, heights5, bottom=heights1+heights2+heights3+heights4, label='Campaign 5')
plt.bar(positions, heights6, bottom=heights1+heights2+heights3+heights4+heights5, label='Last Campaign')
plt.xticks(positions, labels)
plt.xlabel('Month of Enrollment')
plt.ylabel('Number of Accepted Campaigns')
plt.title('Stacked Bar Chart of Accepted Campaigns by Month')
plt.legend()
plt.show()
#1.7 Campaigns 
# List of campaigns
campaigns = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5']
# Calculate the sum of each campaign
campaign_sums = [df[campaign].sum() for campaign in campaigns]
# Define a color palette
colors = ['#ff9999','#f7e6ff','#efccff', '#e7b3ff' , '#df99ff' ,'#d780ff']
# Create a doughnut chart
fig, ax = plt.subplots()
ax.pie(campaign_sums, labels = campaigns, colors = colors, autopct='%1.1f%%', startangle=90, 
pctdistance=0.85)
# Draw a white circle at the center to create the doughnut hole
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal') 
plt.tight_layout()
plt.show()
##############################################################
#B - Etude de la relation entre les attributs :
# 1 3d Scatter Plot nb of total purchases by type of purchases and by age : non sense 
# Assuming 'df' is your DataFrame with the dataset
# Calculate Total Purchases for each customer
df['Total_Purchases'] = df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts']
df['Age'] = 2015 - df['Year_Birth']
# Create a 3D scatter plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
# Scatter plot with Age, Total Purchases, and a marker for each product type
for product in [ 'MntFruits', 'MntMeatProducts', 'MntFishProducts']:
 ax.scatter(df['Age'], df['Total_Purchases'], df[product], label=product)
# Set labels and title
ax.set_xlabel('Age')
ax.set_ylabel('Total Purchases')
ax.set_zlabel('Product Amount')
ax.set_title('3D Scatter Plot of Total Purchases by Age and Product Type')
# Add a legend
ax.legend()
ax.view_init(elev=30, azim=45)
plt.show()
# 1.2 nb of total purchases amount over marital status :
# Defining color palette : 
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
# Calculate the total amount of purchases for each customer by summing up the amount spent on different 
products
df['Total_Purchases'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + 
df['MntSweetProducts'] + df['MntGoldProds']
# Create a box plot of total purchases by marital status using seaborn
sns.boxplot(x='Marital_Status', y='Total_Purchases', data=df, palette=colors)
# Add a gray grid to the plot
plt.grid(axis='y', color='gray', linestyle='--')
# Set the labels and title for the axes and the plot
plt.xlabel('Marital Status')
plt.ylabel('Total Purchases')
plt.title('Box Plot of Total Purchases by Marital Status')
plt.xlabel('Marital Status')
plt.ylabel('Total Purchases')
plt.title('Box Plot of Total Purchases by Marital Status')
plt.show()
# 3 heatmap : for people feautures and total purchase ammount
# we have imported seaborn to plot the heatmap :
# Calculate total purchase amount
df["Total_Purchase"] = df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + df["MntFishProducts"] 
+ df["MntSweetProducts"] + df["MntGoldProds"]
# Calculate correlation matrix
corr = df[["Year_Birth", "Education", "Marital_Status", "Income", "Kidhome", "Teenhome", 
"Total_Purchase"]].corr()
# Plot heatmap
plt.figure(figsize=(10, 10))
purple_palette = sns.color_palette(['#f7e6ff','#efccff', '#e7b3ff' , '#df99ff' ,'#d780ff'])
sns.heatmap(corr, cmap=purple_palette , annot=True, linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()
#5 Catalogue purchase and Total Purchase over 3 years :
# Calculate the total purchases for all channels for each row
df['TotalPurchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
'NumDealsPurchases']].sum(axis=1)
# Convert the date to datetime objects with the correct format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
# Extract the year from the purchase date
df['Year'] = df['Dt_Customer'].dt.year
# Group the data by year and calculate the average total purchases for each year
average_purchase_by_year = df.groupby('Year')['TotalPurchases'].mean()
# Get the individual purchase channels for each year
web_purchases = df.groupby('Year')['NumWebPurchases'].mean()
catalog_purchases = df.groupby('Year')['NumCatalogPurchases'].mean()
store_purchases = df.groupby('Year')['NumStorePurchases'].mean()
# Create an area plot for total purchases and each purchase channel with distinct colors
plt.fill_between(average_purchase_by_year.index, 0, average_purchase_by_year.values, label='Total Purchases', 
color='grey', alpha=0.5)
plt.fill_between(catalog_purchases.index, 0, catalog_purchases.values, label='catalog_purchases', 
color='#d780ff', alpha=0.5)
years = [2012, 2013, 2014]
plt.xticks(years, years)
# Set the x-axis label and title
plt.xlabel('Year')
plt.ylabel('Average Purchases')
plt.title('Average Purchases Over the Years')
# Show a legend
plt.legend()
# Show the plot
plt.show()
# 6 Store and total purchase over three years :
# Calculate the total purchases for all channels for each row
df['TotalPurchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
'NumDealsPurchases']].sum(axis=1)
# Convert the date to datetime objects with the correct format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
# Extract the year from the purchase date
df['Year'] = df['Dt_Customer'].dt.year
# Group the data by year and calculate the average total purchases for each year
average_purchase_by_year = df.groupby('Year')['TotalPurchases'].mean()
# Get the individual purchase channels for each year
web_purchases = df.groupby('Year')['NumWebPurchases'].mean()
catalog_purchases = df.groupby('Year')['NumCatalogPurchases'].mean()
store_purchases = df.groupby('Year')['NumStorePurchases'].mean()
# Create an area plot for total purchases and each purchase channel with distinct colors
plt.fill_between(average_purchase_by_year.index, 0, average_purchase_by_year.values, label='Total Purchases', 
color='grey', alpha=0.5)
plt.fill_between(store_purchases.index, 0, store_purchases.values, label='store_purchases', color='#d780ff', 
alpha=0.5)
years = [2012, 2013, 2014]
plt.xticks(years, years)
# Set the x-axis label and title
plt.xlabel('Year')
plt.ylabel('Average Purchases')
plt.title('Average Purchases Over the Years')
# Show a legend
plt.legend()
# Show the plot
plt.show()
# 7 web and total purchase over three years :
# Calculate the total purchases for all channels for each row
df['TotalPurchases'] = df[['NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
'NumDealsPurchases']].sum(axis=1)
# Convert the date to datetime objects with the correct format
df['Dt_Customer'] = pd.to_datetime(df['Dt_Customer'], format='%d-%m-%Y')
# Extract the year from the purchase date
df['Year'] = df['Dt_Customer'].dt.year
# Group the data by year and calculate the average total purchases for each year
average_purchase_by_year = df.groupby('Year')['TotalPurchases'].mean()
# Get the individual purchase channels for each year
web_purchases = df.groupby('Year')['NumWebPurchases'].mean()
catalog_purchases = df.groupby('Year')['NumCatalogPurchases'].mean()
store_purchases = df.groupby('Year')['NumStorePurchases'].mean()
# Create an area plot for total purchases and each purchase channel with distinct colors
plt.fill_between(average_purchase_by_year.index, 0, average_purchase_by_year.values, label='Total Purchases', 
color='grey', alpha=0.5)
plt.fill_between(web_purchases.index, 0, web_purchases.values, label='web_purchases', color='#d780ff', 
alpha=0.5)
years = [2012, 2013, 2014]
plt.xticks(years, years)
# Set the x-axis label and title
plt.xlabel('Year')
plt.ylabel('Average Purchases')
plt.title('Average Purchases Over the Years')
# Show a legend
plt.legend()
# Show the plot
plt.show()
#CCCCCCCCCC- Etude du Salaire 
#1 avg income by education 
sns.set_style('darkgrid')
# Group data by education and calculate mean income
grouped = df.groupby("Education")["Income"].mean()
# Sort grouped data by income values in ascending order
grouped = grouped.sort_values()
# Create labels, heights, and colors for bar chart
labels = grouped.index
heights = grouped.values
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
# Plot bar chart
plt.bar(labels, heights, color=colors)
plt.xlabel("Education")
plt.ylabel("Average Income")
plt.title("Bar Chart of Average Income by Education")
# Loop over the bars and add text
for i, bar in enumerate(plt.gca().patches):
 # Get the x and y coordinates of the bar
 x = bar.get_x() + bar.get_width() / 2
 y = bar.get_height()
 # Format the height as a string with two decimal places
 height = f"{y:.2f}"
 # Annotate the bar with the height
 plt.text(x, y, height, ha="center", va="bottom")
plt.show()
#2.2 avg of income distributed on diff marital status for people without the absurd category:
# Filter data by year of birth
df = df.query("Year_Birth <= 1997")
# Filter data by marital status
df = df.query("Marital_Status != 'Absurd'")
# Group data by marital status and calculate mean income
grouped = df.groupby("Marital_Status")["Income"].mean()
# Sort grouped data by income values in ascending order
grouped = grouped.sort_values()
# Create labels, heights, and colors for bar chart
labels = grouped.index
heights = grouped.values
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
# Plot bar chart
plt.bar(labels, heights, color=colors)
plt.xlabel("Marital_Status")
plt.ylabel("Average Income")
plt.title("Bar Chart of Average Income by Marital status")
plt.show()
#2.3 avg of income distributed on by age 
# Calculate the age of each customer based on their year of birth
df['Age'] = 2015 - df['Year_Birth']
# Define age categories and labels
age_bins = [0, 30, 40, 50, 60, 100] # Define age bins
age_labels = ["18-30", "31-40", "41-50", "51-60", "61+"]
df['Age_Category'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
# Group data by age category and calculate the mean income
grouped = df.groupby("Age_Category")["Income"].mean()
# Create labels, heights, and colors for the bar chart
labels = grouped.index
heights = grouped.values
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
# Plot bar chart
plt.bar(labels, heights, color=colors)
plt.xlabel("Age Category")
plt.ylabel("Average Income")
plt.title("Bar Chart of Average Income by Age Category")
plt.show()
# 2.4 nb of total purchases amount over income :
# Filter the data
df = df[(df['Income'] <= 170000) & (df['Income'] >= 1000)]
# Calculate the total amount of purchases for each customer by summing up the amount spent on different 
products
df['Total_Purchases'] = df['MntWines'] + df['MntFruits'] + df['MntMeatProducts'] + df['MntFishProducts'] + 
df['MntSweetProducts'] + df['MntGoldProds']
# Create a scatter plot of income vs total purchases using matplotlib
plt.scatter(df['Income'], df['Total_Purchases'])
plt.xlabel('Income')
plt.ylabel('Total Purchases')
plt.title('Scatter Plot of Income vs Total Purchases')
sns.regplot(x='Income', y='Total_Purchases', data=df , color='#d780ff')
plt.show()
#D Disecting the population :
"""
# Calculate the mean and standard deviation
mean_income = np.mean(df['Income'])
std_dev = np.std(df['Income'])
# Define the cutoffs for low, medium, and high based on standard deviation
low_cutoff = mean_income - std_dev
medium_cutoff = mean_income + std_dev
high_cutoff = mean_income + (2 * std_dev)
# Define the bins for categorizing income
bins = [-np.inf, low_cutoff, medium_cutoff, high_cutoff, np.inf]
# Define labels for the categories
labels = ['Very Low', 'Low', 'Medium', 'High']
# Create the "Income Category" column
df['Income_Category'] = pd.cut(df['Income'], bins=bins, labels=labels)
df['Age'] = 2015 - df['Year_Birth']
"""
# Income_Ctageory Distribution :
# Create a figure with a custom size and resolution
plt.figure(figsize=(10, 6), dpi=100)
# Filter the data to remove outliers
df = df[(df['Income'] <= 170000) & (df['Income'] >= 1000)]
# Count the occurrences of each Income_Category
income_counts = df['Income_Category'].value_counts().sort_values()
# Create a bar chart of the count distribution for each Income_Category
plt.bar(income_counts.index, income_counts, color="#efccff", edgecolor="white")
# Add lollipops as scatter points
plt.scatter(income_counts.index, income_counts, color="#ae00ff", marker='o', s=50)
# Add labels and a title to the plot
plt.xlabel("Income Category")
plt.ylabel("Count")
plt.title("Lollipop Chart of Income_Category Distribution")
# Set the x-axis ticks for a clearer result
plt.xticks(rotation='vertical')
# Add vertical grid lines for better visibility
plt.grid(axis='y', linestyle='--', alpha=0.6)
# Show the plot
plt.show()
#E1: Distribution of population for high_income :
#1- Age Distributon
# Filter the data to keep only the rows where the income category is high
df = df.loc[df["Income_Category"] == "High"]
# Create a figure with a custom size and resolution
plt.figure(figsize=(10, 6), dpi=100)
# Create a histogram of age for the high-income population
plt.hist(df["Age"], bins=range(0, max(df["Age"]) + 5, 5), color="#ae00ff", edgecolor="white")
# Add labels and a title to the plot
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution for High-Income Population")
# Set the x-axis ticks for a clearer result
plt.xticks(range(0, max(df["Age"]) + 1, 5))
# Add vertical grid lines for better visibility
plt.grid(axis='x', linestyle='--', alpha=0.6)
# Show the plot
plt.show()
#2= Donut Chart for education distribution :
# education for high income population 
sns.set_style("darkgrid")
# Filter the DataFrame for "High" income category
df = df.loc[df["Income_Category"] == "High"]
# Count the frequency of education level
counts = df["Education"].value_counts()
# Create a pie chart of education level
plt.pie(counts, labels=counts.index, colors=["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff"] , shadow=True)
# Add a circle at the center of the pie chart
circle = plt.Circle((0, 0), 0.5, color="white")
ax = plt.gca()
ax.add_artist(circle)
# Add percentage labels and a title to the donut chart
plt.xlabel("Education Level")
plt.title("Distribution of Education Level for Medium Income Population")
# Calculate and annotate with percentage labels
total = sum(counts)
percentages = [(count / total) * 100 for count in counts]
labels = [f"{edu} ({percent:.1f}%)" for edu, percent in zip(counts.index, percentages)]
plt.legend(labels, title="Education Level", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
#3 Countplot for Marital status for high income category ( state the difference between a counterplot bar and 
histo
df = df.loc[df["Income_Category"] == "High"]
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
# Create a count plot of marital status for high-income population
sns.countplot(x=df["Marital_Status"], palette=colors)
plt.xlabel("Marital Status")
plt.ylabel("Frequency")
plt.title("Count Plot of Marital Status for High-Income Population")
plt.show()
# 4 Radar Chart of products distribution for High Income Population 
# Filter the data
df = df.loc[(df["Marital_Status"] == "Together") & (df["Age"] >= 35) & (df["Age"] <= 40) & 
(df["Education"].isin(["PhD", "Graduation"])&(df["Income_Category"] == "High"))]
# Select the columns for the products and calculate the mean by row
products = df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", 
"MntGoldProds"]].mean(axis=0)
# Create a radar chart for the average of product consumption
theta = np.linspace(0, 2*np.pi, len(products), endpoint=False)
plt.polar(theta, products, color="#e7b3ff", marker='o') # Use specified color and add markers
plt.fill(theta, products, color="#e7b3ff", alpha=0.3) # Fill the radar area with alpha for transparency
# Add labels and a title to the plot
plt.thetagrids(np.degrees(theta), products.index)
plt.title("Avg Consumption by Product")
# Show or save the plot
plt.show()
# 5 Distribution of acceptance rate 
df = df.loc[(df["Marital_Status"] == "Together") & (df["Age"] >= 35) & (df["Age"] <= 40) & 
(df["Education"].isin(["PhD", "Graduation"])&(df["Income_Category"] == "High"))]
# Calculate the acceptance rate for each campaign by dividing the mean of accepted customers by the total 
number of customers
acceptance_rate = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
"AcceptedCmp5" , "Response"]].mean()
# Plot the horizontal bar chart with labels and title
plt.barh(acceptance_rate.index, acceptance_rate.values * 100 , color = "#efccff")
plt.ylabel("Campaign")
plt.xlabel("Acceptance Rate (%)")
plt.title("Acceptance Rate for Each Campaign in the Population")
plt.show()
#6 Channels for high income population
df = df.loc[(df["Marital_Status"] == "Together") & (df["Age"] >= 35) & (df["Age"] <= 40) & 
(df["Education"].isin(["PhD", "Graduation"])) & (df["Income_Category"] == "High")]
colors = ["#f7e6ff", "#efccff", "#e7b3ff"]
# Calculate the total number of purchases for each channel by summing up the columns
channels = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum()
# Plot the pie chart with labels and title
plt.pie(channels.values, labels=channels.index, autopct="%1.1f%%" , shadow = True , colors = colors , 
explode=(0,0.2,0))
plt.title("Percentage of People that Use Each Channel in the Population")
plt.show()
# E2 : Population with medium income :
#1- Age Distributon
# Filter the data to keep only the rows where the income category is high
df = df.loc[df["Income_Category"] == "Medium"]
# Create a figure with a custom size and resolution
plt.figure(figsize=(10, 6), dpi=100)
# Create a histogram of age for the high-income population
plt.hist(df["Age"], bins=range(0, max(df["Age"]) + 5, 5), color="#efccff", edgecolor="white")
# Add labels and a title to the plot
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution for Medium-Income Population")
# Set the x-axis ticks for a clearer result
plt.xticks(range(0, max(df["Age"]) + 1, 5))
# Add vertical grid lines for better visibility
plt.grid(axis='x', linestyle='--', alpha=0.6)
# Show the plot
plt.show()
#2= Donut Chart for education distribution :
# education for high income population 
sns.set_style("darkgrid")
# Filter the DataFrame for "High" income category
df = df.loc[df["Income_Category"] == "Medium"]
# Count the frequency of education level
counts = df["Education"].value_counts()
# Create a pie chart of education level
plt.pie(counts, labels=counts.index, colors=["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff"] , shadow=True)
# Add a circle at the center of the pie chart
circle = plt.Circle((0, 0), 0.5, color="white")
ax = plt.gca()
ax.add_artist(circle)
# Add percentage labels and a title to the donut chart
plt.xlabel("Education Level")
plt.title("Distribution of Education Level for Medium Income Population")
# Calculate and annotate with percentage labels
total = sum(counts)
percentages = [(count / total) * 100 for count in counts]
labels = [f"{edu} ({percent:.1f}%)" for edu, percent in zip(counts.index, percentages)]
plt.legend(labels, title="Education Level", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
#3 Countplot for Marital status for Medium income category ( state the difference between a counterplot bar 
and histo
df = df.loc[df["Income_Category"] == "Medium"]
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
# Create a count plot of marital status for high-income population
sns.countplot(x=df["Marital_Status"], palette=colors)
plt.xlabel("Marital Status")
plt.ylabel("Frequency")
plt.title("Count Plot of Marital Status for Medium-Income Population")
plt.show()
# 4 Radar Chart of products distribution for Medium Income Population 
# Filter the data
df = df.loc[df["Income_Category"] == "Medium"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 55) & (df["Age"] <= 60) & 
(df["Education"].isin(["Graduation"]))]
# Select the columns for the products and calculate the mean by row
products = df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", 
"MntGoldProds"]].mean(axis=0)
# Create a radar chart for the average of product consumption
theta = np.linspace(0, 2*np.pi, len(products), endpoint=False)
plt.polar(theta, products, color="#e7b3ff", marker='o') # Use specified color and add markers
plt.fill(theta, products, color="#e7b3ff", alpha=0.3) # Fill the radar area with alpha for transparency
# Add labels and a title to the plot
plt.thetagrids(np.degrees(theta), products.index)
plt.title("Avg Consumption by Product")
# Show or save the plot
plt.show()
# 5 Distribution of acceptance rate 
df = df.loc[df["Income_Category"] == "Medium"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 55) & (df["Age"] <= 60) & 
(df["Education"].isin(["Graduation"]))]
# Calculate the acceptance rate for each campaign by dividing the mean of accepted customers by the total 
number of customers
acceptance_rate = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
"AcceptedCmp5" , "Response"]].mean()
# Plot the horizontal bar chart with labels and title
plt.barh(acceptance_rate.index, acceptance_rate.values * 100 , color = "#efccff")
plt.ylabel("Campaign")
plt.xlabel("Acceptance Rate (%)")
plt.title("Acceptance Rate for Each Campaign in the Population")
plt.show()
#6 Channels for Medium income population
df = df.loc[df["Income_Category"] == "Medium"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 55) & (df["Age"] <= 60) & 
(df["Education"].isin(["Graduation"]))]
colors = ["#f7e6ff", "#efccff", "#e7b3ff"]
# Calculate the total number of purchases for each channel by summing up the columns
channels = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum()
# Plot the pie chart with labels and title
plt.pie(channels.values, labels=channels.index, autopct="%1.1f%%" , shadow = True , colors = colors , 
explode=(0,0,0.2))
plt.title("Percentage of People that Use Each Channel in the Population")
plt.show()
#E3 - Low Income Study :
#1- Age Distributon
# Filter the data to keep only the rows where the income category is high
df = df.loc[df["Income_Category"] == "Low"]
# Create a figure with a custom size and resolution
plt.figure(figsize=(10, 6), dpi=100)
# Create a histogram of age for the high-income population
plt.hist(df["Age"], bins=range(0, max(df["Age"]) + 5, 5), color="#efccff", edgecolor="white")
# Add labels and a title to the plot
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution for Low-Income Population")
# Set the x-axis ticks for a clearer result
plt.xticks(range(0, max(df["Age"]) + 1, 5))
# Add vertical grid lines for better visibility
plt.grid(axis='x', linestyle='--', alpha=0.6)
# Show the plot
plt.show()
#2= Donut Chart for education distribution :
# education for high income population 
sns.set_style("darkgrid")
df = df.loc[df["Income_Category"] == "Low"]
# Count the frequency of education level
counts = df["Education"].value_counts()
# Create a pie chart of education level
plt.pie(counts, labels=counts.index, colors=["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff"] , shadow=True)
# Add a circle at the center of the pie chart
circle = plt.Circle((0, 0), 0.5, color="white")
ax = plt.gca()
ax.add_artist(circle)
plt.xlabel("Education Level")
plt.title("Distribution of Education Level for low Income Population")
total = sum(counts)
percentages = [(count / total) * 100 for count in counts]
labels = [f"{edu} ({percent:.1f}%)" for edu, percent in zip(counts.index, percentages)]
plt.legend(labels, title="Education Level", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
#3 Countplot for Marital status for Medium income category ( state the difference between a counterplot bar 
and histo
df = df.loc[df["Income_Category"] == "Low"]
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
sns.countplot(x=df["Marital_Status"], palette=colors)
plt.xlabel("Marital Status")
plt.ylabel("Frequency")
plt.title("Count Plot of Marital Status for Low-Income Population")
plt.show()
# 4 Radar Chart of products distribution for low Income Population 
# Filter the data
df = df.loc[df["Income_Category"] == "Low"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 40) & (df["Age"] <= 45) & 
(df["Education"].isin(["Graduation"]))]
products = df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", 
"MntGoldProds"]].mean(axis=0)
theta = np.linspace(0, 2*np.pi, len(products), endpoint=False)
plt.polar(theta, products, color="#e7b3ff", marker='o') # Use specified color and add markers
plt.fill(theta, products, color="#e7b3ff", alpha=0.3) # Fill the radar area with alpha for transparency
plt.thetagrids(np.degrees(theta), products.index)
plt.title("Avg Consumption by Product")
# Show or save the plot
plt.show()
# 5 Distribution of acceptance rate 
df = df.loc[df["Income_Category"] == "Low"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 40) & (df["Age"] <= 45) & 
(df["Education"].isin(["Graduation"]))]
acceptance_rate = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
"AcceptedCmp5" , "Response"]].mean()
plt.barh(acceptance_rate.index, acceptance_rate.values * 100 , color = "#efccff")
plt.ylabel("Campaign")
plt.xlabel("Acceptance Rate (%)")
plt.title("Acceptance Rate for Each Campaign in the Population")
plt.show()
#6 Channels for high income population
df = df.loc[df["Income_Category"] == "Low"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 40) & (df["Age"] <= 45) & 
(df["Education"].isin(["Graduation"]))]
colors = ["#f7e6ff", "#efccff", "#e7b3ff"]
channels = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum()
plt.pie(channels.values, labels=channels.index, autopct="%1.1f%%" , shadow = True , colors = colors , 
explode=(0,0,0.2))
plt.title("Percentage of People that Use Each Channel in the Population")
plt.show()
#E4 - Population with Very-Low Income :
#1- Age Distributon
df = df.loc[df["Income_Category"] == "Very Low"]
plt.figure(figsize=(10, 6), dpi=100)
plt.hist(df["Age"], bins=range(0, max(df["Age"]) + 5, 5), color="#efccff", edgecolor="white")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution for Very Low-Income Population")
plt.xticks(range(0, max(df["Age"]) + 1, 5))
plt.grid(axis='x', linestyle='--', alpha=0.6)
plt.show()
#2= Donut Chart for education distribution :
sns.set_style("darkgrid")
df = df.loc[df["Income_Category"] == "Very Low"]
counts = df["Education"].value_counts()
plt.pie(counts, labels=counts.index, colors=["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff"] , shadow=True)
circle = plt.Circle((0, 0), 0.5, color="white")
ax = plt.gca()
ax.add_artist(circle)
plt.xlabel("Education Level")
plt.title("Distribution of Education Level for Very Low Income Population")
total = sum(counts)
percentages = [(count / total) * 100 for count in counts]
labels = [f"{edu} ({percent:.1f}%)" for edu, percent in zip(counts.index, percentages)]
plt.legend(labels, title="Education Level", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
plt.show()
#3 Countplot for Marital status for Medium income category ( state the difference between a counterplot bar 
and histo
df = df.loc[df["Income_Category"] == "Very Low"]
colors = ["#f7e6ff", "#efccff", "#e7b3ff", "#df99ff", "#d780ff"]
sns.countplot(x=df["Marital_Status"], palette=colors)
plt.xlabel("Marital Status")
plt.ylabel("Frequency")
plt.title("Count Plot of Marital Status for Very Low-Income Population")
plt.show()
# 4 Radar Chart of products distribution for High Income Population 
# Filter the data
df = df.loc[df["Income_Category"] == "Very Low"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 35) & (df["Age"] <= 40) & 
(df["Education"].isin(["Graduation"]))]
products = df[["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", 
"MntGoldProds"]].mean(axis=0)
theta = np.linspace(0, 2*np.pi, len(products), endpoint=False)
plt.polar(theta, products, color="#e7b3ff", marker='o') # Use specified color and add markers
plt.fill(theta, products, color="#e7b3ff", alpha=0.3) # Fill the radar area with alpha for transparency
plt.thetagrids(np.degrees(theta), products.index)
plt.title("Avg Consumption by Product")
# Show or save the plot
plt.show()
# 5 Distribution of acceptance rate 
df = df.loc[df["Income_Category"] == "Very Low"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 35) & (df["Age"] <= 40) & 
(df["Education"].isin(["Graduation"]))]
acceptance_rate = df[["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", 
"AcceptedCmp5" , "Response"]].mean()
plt.barh(acceptance_rate.index, acceptance_rate.values * 100 , color = "#efccff")
plt.ylabel("Campaign")
plt.xlabel("Acceptance Rate (%)")
plt.title("Acceptance Rate for Each Campaign in the Population")
plt.show()
#6 Channels for high income population
df = df.loc[df["Income_Category"] == "Very Low"]
df = df.loc[(df["Marital_Status"] == "Married") & (df["Age"] >= 35) & (df["Age"] <= 40) & 
(df["Education"].isin(["Graduation"]))]
colors = ["#f7e6ff", "#efccff", "#e7b3ff"]
channels = df[["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]].sum()
plt.pie(channels.values, labels=channels.index, autopct="%1.1f%%" , shadow = True , colors = colors , 
explode=(0,0,0.2))
plt.title("Percentage of People that Use Each Channel in the Population")
plt.show()
""