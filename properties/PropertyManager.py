import pandas as pd
import csv
import random

class Property:
    def __init__(self, property_id, location, ptype, nightly_price, features, tags):
        self.property_id = property_id
        self.location = location
        self.ptype = ptype
        self.nightly_price = nightly_price
        self.features = features
        self.tags = tags

# randomly generate [count] number of properties, and write into properties.csv file
def generate_properties_csv(filename="properties.csv", count=100):   
    location_list = ["Paris", "London", "Rome"]
    ptype_list = ["cabin", "condo", "house"]
    features_list = ["hot tub", "WiFi", "pet friendly"]
    tags_list = ["remote", "family-friendly", "nightlife", "mountain", "beach"]
    properties = []
    for i in range(1, count + 1):
        properties.append(
            Property(
                property_id=i,
                location=random.choice(location_list),
                ptype=random.choice(ptype_list),
                nightly_price=random.randint(30, 100) * 5,
                features=random.sample(features_list, random.randint(1, 3)),
                tags=random.sample(tags_list, random.randint(2, 4))
            )
        )

    with open(filename, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["location", "type", "nightly_price", "features", "tags"])
        for prop in properties:
            writer.writerow([
                prop.location,
                prop.ptype,
                prop.nightly_price,
                ", ".join(prop.features),
                ", ".join(prop.tags)
            ])


class PropertyManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.properties = pd.read_csv(filepath)

    def display_properties(self):
        print("\nAvailable Properties:")
        print(self.properties[['location', 'type', 'nightly_price', 'features']])
