# Sample data for budget categories and their proportions


# Category labels

import matplotlib.pyplot as plt
import numpy as np
items = ["Rent", "Groceries", "Utilities", "Transportation", "Entertainment", "Savings"]
data = [35, 20, 15, 10, 10, 10]  # Percentages can be adjusted
plt.pie(data,labels=items)
plt.pie(data, labels=items, autopct="%.2f%%")
plt.title("Monthly_Expenses")
# plt.show()
plt.savefig("generatedImage.png")
