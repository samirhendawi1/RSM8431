class User:
    def __init__(self, user_id, name, group_size, environment, budget_min,
                 budget_max, travel_dates=None):
        self.user_id = user_id
        self.name = name
        self.group_size = group_size
        self.environment = environment
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.travel_dates = travel_dates


class UserManager:
    def __init__(self):
        self.users = {}
        self.current_user_id = None

    def create_user(self):
        try:
            user_id = input("Enter user ID: ")
            name = input("Enter name: ")
            group_size = int(input("Enter group size: "))
            environment = input("Preferred environment (mountain/lake/beach/city): ")
            budget_min = float(input("Enter minimum budget: "))
            budget_max = float(input("Enter maximum budget: "))
            travel_dates = input("Enter travel dates (optional): ") or None
        except:
            print("Invalid input. Please try again.")
            return

        user = User(user_id, name, group_size, environment, budget_min, budget_max, travel_dates)
        self.users[user_id] = user
        self.current_user_id = user_id
        print("User created successfully.")

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
        user.travel_dates = input(f"Travel dates ({user.travel_dates}): ") or user.travel_dates
        print("Profile updated.")

    def get_current_user(self):
        return self.users.get(self.current_user_id)
