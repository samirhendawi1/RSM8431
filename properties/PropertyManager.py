import pandas as pd
import csv
import random
import os
from pathlib import Path

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
    # Ensure target folder exists
    try:
        p = Path(filename)
        if p.parent and str(p.parent) not in ("", "."):
            p.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        # If we can't create dirs, let the write attempt raise a clear error
        pass

    location_list = ["Paris", "London", "Rome"]
    ptype_list = ["cabin", "condo", "house"]
    features_list = ["hot tub", "WiFi", "pet friendly"]
    tags_list = ["remote", "family-friendly", "nightlife", "mountain", "beach"]

    properties = []
    for i in range(1, int(count) + 1 if isinstance(count, int) else 101):
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
    random.seed(42)

    # Write CSV defensively
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
        try:
            self.properties = pd.read_csv(filepath)
        except Exception:
            # If read fails, initialize an empty but schema-compatible DataFrame
            self.properties = pd.DataFrame(columns=["location", "type", "nightly_price", "features", "tags"])

        # Normalize expected columns/types so downstream code is safe
        for col in ["location", "type", "features", "tags"]:
            if col not in self.properties.columns:
                self.properties[col] = ""
        if "nightly_price" not in self.properties.columns:
            self.properties["nightly_price"] = 0

        # Coerce to safe dtypes
        try:
            self.properties["nightly_price"] = pd.to_numeric(self.properties["nightly_price"], errors="coerce").fillna(0)
        except Exception:
            pass

        # Ensure features/tags are strings (CSV stores comma-separated lists as strings)
        for col in ["features", "tags"]:
            try:
                self.properties[col] = self.properties[col].fillna("").astype(str)
            except Exception:
                self.properties[col] = ""

    def display_properties(self):
        print("\nAvailable Properties:")
        wanted = ["location", "type", "nightly_price", "features"]
        available = [c for c in wanted if c in self.properties.columns]
        if available:
            print(self.properties[available])
        else:
            # Fallback: show whatever we have
            print(self.properties.head())
