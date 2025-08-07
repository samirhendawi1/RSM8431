import pandas as pd


class PropertyManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.properties = pd.read_csv(filepath)

    def display_properties(self):
        print("\nAvailable Properties:")
        print(self.properties[['location', 'type', 'nightly_price', 'features']])
