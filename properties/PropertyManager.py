import pandas as pd, csv, random, os
from pathlib import Path

class PropertyManager:
    def __init__(self, filepath: str):
        self.filepath = filepath
        try:
            self.properties = pd.read_csv(filepath)
        except Exception:
            self.properties = pd.DataFrame(columns=[
                "property_id","location","environment","property_type",
                "nightly_price","features","tags","min_guests","max_guests"
            ])
        self._normalize()

    def _normalize(self):
        df = self.properties
        df.columns = [str(c).strip() for c in df.columns]

        # Ensure columns
        for c in ["property_id","location","environment","property_type",
                  "nightly_price","features","tags","min_guests","max_guests"]:
            if c not in df.columns:
                df[c] = 0 if c in ("nightly_price","min_guests","max_guests","property_id") else ""

        # Types
        df["nightly_price"] = pd.to_numeric(df["nightly_price"], errors="coerce").fillna(0.0)
        for c in ["min_guests","max_guests","property_id"]:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
        for c in ["features","tags","location","environment","property_type"]:
            df[c] = df[c].fillna("").astype(str)

        # Fulltext
        df["__fulltext__"] = (
                df["location"] + " " + df["environment"] + " " + df["property_type"] + " " + df["features"] + " " + df["tags"]
        ).str.lower()
        self.properties = df

    def display_properties(self):
        cols = ["property_id","location","environment","property_type","nightly_price",
                "min_guests","max_guests","features","tags"]
        cols = [c for c in cols if c in self.properties.columns]
        print("\nAvailable Properties:")
        print(self.properties[cols].head(50).to_string(index=False))


def generate_properties_csv(filename="data/properties_with_capacity_types.csv", count=140):
    """Optional utility: generates a dataset with environment + property_type + capacity."""
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    locations = ["Toronto","Vancouver","Montreal","Calgary","Ottawa","New York","Los Angeles","Miami","Austin","Seattle","Paris","London","Rome","Barcelona","Lisbon"]
    environments = ["beach","mountain","city","lake","desert","forest","island","suburban"]
    property_types = ["condo","townhouse","detached","semi-detached","loft","studio","villa","cabin"]
    features_pool = ["WiFi","hot tub","pool","gym","pet friendly","kitchen","parking","AC","washer","dryer","EV charger"]
    tags_pool = ["family-friendly","remote","nightlife","romantic","surfing","skiing","quiet","downtown","luxury","budget","scenic","hiking","business","student"]

    random.seed(11)
    rows = []
    for i in range(1, int(count)+1):
        loc = random.choice(locations)
        env = random.choice(environments)
        ptype = random.choice(property_types)
        price = random.randint(60, 1000)
        kf = random.randint(2,5); kt = random.randint(2,5)
        features = ", ".join(sorted(set(random.sample(features_pool, kf))))
        tags = ", ".join(sorted(set(random.sample(tags_pool+[env, ptype], kt))))
        min_g = random.randint(1,4)
        max_g = min(min_g + random.randint(1,7), 14)
        rows.append([i, loc, env, ptype, price, features, tags, min_g, max_g])

    with open(filename, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["property_id","location","environment","property_type","nightly_price","features","tags","min_guests","max_guests"])
        w.writerows(rows)
    return filename
