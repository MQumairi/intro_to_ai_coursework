# %%
import pandas as pd
import matplotlib.pyplot as plt

# %%
data = pd.read_csv("dataset.csv")
# data.count()

list_of_cols_to_keep = ["Date", "RegionName", "AveragePrice"]
data_region_price = data[list_of_cols_to_keep]

# data_region_price.count()

london_data = data_region_price[data_region_price["RegionName"]
                                == "City of London"]

manchester_data = data_region_price[data_region_price["RegionName"]
                                    == "Manchester"]

birmingham_data = data_region_price[data_region_price["RegionName"]
                                    == "Birmingham"]

leeds_data = data_region_price[data_region_price["RegionName"]
                               == "Leeds"]
liverpool_data = data_region_price[data_region_price["RegionName"]
                                   == "Liverpool"]

bradford_data = data_region_price[data_region_price["RegionName"]
                                  == "Bradford"]

sheffield_data = data_region_price[data_region_price["RegionName"]
                                   == "Sheffield"]

plt.figure(figsize=[10, 35])
plt.subplot(7, 1, 1)
plt.plot(london_data["Date"], london_data["AveragePrice"])
plt.subplot(7, 1, 2)
plt.plot(manchester_data["Date"], manchester_data["AveragePrice"])
plt.subplot(7, 1, 3)
plt.plot(birmingham_data["Date"], birmingham_data["AveragePrice"])
plt.subplot(7, 1, 4)
plt.plot(leeds_data["Date"], leeds_data["AveragePrice"])
plt.subplot(7, 1, 5)
plt.plot(liverpool_data["Date"], liverpool_data["AveragePrice"])
plt.subplot(7, 1, 6)
plt.plot(bradford_data["Date"], bradford_data["AveragePrice"])
plt.subplot(7, 1, 7)
plt.plot(sheffield_data["Date"], sheffield_data["AveragePrice"])

# %%
