import os
import csv
class User:
    def __init__(self, user_id, name, group_size, environment, budget_min,
                 budget_max):
        self.user_id = user_id
        self.name = name
        self.group_size = group_size
        self.environment = environment
        self.budget_min = budget_min
        self.budget_max = budget_max


class UserManager:
    def __init__(self, csv_file="data/users.csv"):
        self.users = {}
        self.current_user_id = None
        self.csv_file = csv_file
        self.load_from_csv()  # Load automatically once started

    def save_to_csv(self):
        with open(self.csv_file, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "name", "group_size", "environment", "budget_min", "budget_max"])
            for user_id, user in self.users.items():
                writer.writerow([user.user_id, user.name, user.group_size, user.environment,
                                 user.budget_min, user.budget_max])


    def load_from_csv(self):
        try:
            with open(self.csv_file, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    user = User(
                        row["user_id"],
                        row["name"],
                        int(row["group_size"]),
                        row["environment"],
                        float(row["budget_min"]),
                        float(row["budget_max"])
                    )
                    self.users[user.user_id] = user
        except FileNotFoundError:
            pass


    def create_user(self):
        try:
            user_id = input("Enter user ID: ")
            name = input("Enter name: ")
            group_size = int(input("Enter group size: "))
            environment = input("Preferred environment (mountain/lake/beach/city): ")
            budget_min = float(input("Enter minimum budget: "))
            budget_max = float(input("Enter maximum budget: "))
        except:
            print("Invalid input. Please try again.")
            return

        user = User(user_id, name, group_size, environment, budget_min, budget_max)
        self.users[user_id] = user
        self.current_user_id = user_id
        print("User created successfully.")
        self.save_to_csv()  # Automatically save to csv_file

    def edit_profile(self):
        if self.current_user_id is None:
            print("No user selected.")
            return

        user = self.users[self.current_user_id]
        print("Editing profile for:", user.name)
        user.name = input(f"Name ({user.name}): ") or user.name
        user.group_size = int(input(f"Group size ({user.group_size}): ") or user.group_size)
        user.environment = input(f"Environment ({user.environment}): ") or user.environment
        user.budget_min = float(input(f"Minimum budget ({user.budget_min}): ") or user.budget_min)
        user.budget_max = float(input(f"Maximum budget ({user.budget_max}): ") or user.budget_max)
        print("Profile updated.")
        self.save_to_csv()  # Automatically save to csv_file


    def get_current_user(self):
        return self.users.get(self.current_user_id)

    def delete_user(self, user_id):
        if user_id in self.users:
            del self.users[user_id]
            if self.current_user_id == user_id:
                self.current_user_id = None
            self.save_to_csv()
            print(f"User {user_id} deleted and CSV updated.")
            return True
        else:
            print(f"User {user_id} not found.")
            return False
