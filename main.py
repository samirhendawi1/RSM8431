from user.UserManager import UserManager
from properties.PropertyManager import PropertyManager, generate_properties_csv
from recommender.Recommender import Recommender
# Keep a single import path for LLMHelper.
from recommender.llm import LLMHelper


def _print_profile(user):
    print("\n--- Current User Profile ---")
    print(f"ID:            {getattr(user, 'user_id', 'N/A')}")
    print(f"Name:          {getattr(user, 'name', 'N/A')}")
    print(f"Group size:    {getattr(user, 'group_size', 'N/A')}")
    print(f"Environment:   {getattr(user, 'environment', 'N/A')}")
    print(f"Budget min:    {getattr(user, 'budget_min', 'N/A')}")
    print(f"Budget max:    {getattr(user, 'budget_max', 'N/A')}")
    print(f"Travel dates:  {getattr(user, 'travel_dates', 'N/A')}")
    print("-----------------------------\n")


def main():
    # Generate property list (as your updated main does)
    generate_properties_csv("data/properties.csv", 100)

    user_manager = UserManager()
    property_manager = PropertyManager('data/properties.csv')
    recommender = Recommender()
    # Use expanded CSV if present; otherwise LLMHelper will auto-detect.
    llm = LLMHelper(csv_path="data/properties_expanded.csv")

    while True:
        print("\nMenu:")
        print("1. Create User")
        print("2. Edit Profile")
        print("3. View Profile")           # NEW
        print("4. Show Properties")
        print("5. Get Recommendations")
        print("6. LLM Summary")
        print("7. Delete Profile")          # NEW
        print("8. Exit")

        choice = input("Choose an option: ").strip()

        if choice == '1':
            if hasattr(user_manager, 'create_user'):
                user_manager.create_user()
            else:
                print("Create user is not available in UserManager.")
        elif choice == '2':
            if hasattr(user_manager, 'edit_profile'):
                user_manager.edit_profile()
            else:
                print("Edit profile is not available in UserManager.")
        elif choice == '3':  # View Profile
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if user:
                _print_profile(user)
            else:
                print("No user selected.")
        elif choice == '4':
            if hasattr(property_manager, 'display_properties'):
                property_manager.display_properties()
            else:
                print("Property display is not available.")
        elif choice == '5':
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if user is None:
                print("No user selected.")
                continue
            try:
                df = property_manager.properties
                recs = recommender.recommend(user, df)
                print("\nTop Recommendations:")
                try:
                    from tabulate import tabulate
                    print(tabulate(recs, headers='keys', tablefmt='github', showindex=False))
                except Exception:
                    print(recs)
            except Exception as e:
                print("Error generating recommendations:", e)
        elif choice == '6':
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if user:
                prompt = (
                    f"Write a fun intro for a user looking for a {user.environment} stay "
                    f"with {user.group_size} friends under ${user.budget_max}/night."
                )
                print("\nLLM Response:")
                print(llm.generate_travel_blurb(prompt))
            else:
                print("No user selected.")
        elif choice == '7':  # Delete Profile
            user = user_manager.get_current_user() if hasattr(user_manager, 'get_current_user') else None
            if not user:
                print("No user selected.")
                continue
            confirm = input(f"Type DELETE to remove profile '{getattr(user, 'name', '')}': ").strip()
            if confirm != "DELETE":
                print("Cancelled.")
                continue
            try:
                uid = getattr(user, 'user_id', None)
                if hasattr(user_manager, 'users') and isinstance(user_manager.users, dict):
                    if uid in user_manager.users:
                        del user_manager.users[uid]
                    else:
                        for k, v in list(user_manager.users.items()):
                            if v is user:
                                del user_manager.users[k]
                if getattr(user_manager, 'current_user_id', None) == uid:
                    user_manager.current_user_id = None
                print("Profile deleted.")
            except Exception as e:
                print("Failed to delete profile:", e)
        elif choice == '8':
            break
        else:
            print("Invalid option.")


if __name__ == '__main__':
    main()
