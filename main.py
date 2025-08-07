from user.UserManager import UserManager
from properties.PropertyManager import PropertyManager
from recommender.Recommender import Recommender
from recommender.llm import LLMHelper


def main():
    user_manager = UserManager()
    property_manager = PropertyManager('data/properties.csv')
    recommender = Recommender()
    llm = LLMHelper()

    while True:
        print("\nMenu:")
        print("1. Create User")
        print("2. Edit Profile")
        print("3. View Properties")
        print("4. Get Recommendations")
        print("5. Get LLM Summary")
        print("6. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            user_manager.create_user()
        elif choice == '2':
            user_manager.edit_profile()
        elif choice == '3':
            property_manager.display_properties()
        elif choice == '4':
            user = user_manager.get_current_user()
            if user:
                recommendations = recommender.recommend(user, property_manager.properties)
                print("\nTop Recommendations:")
                print(recommendations[['location', 'type', 'nightly_price', 'features', 'fit_score']])
            else:
                print("No user selected.")
        elif choice == '5':
            user = user_manager.get_current_user()
            if user:
                prompt = f"Write a fun intro for a user looking to travel to a {user.environment} cabin with {user.group_size} friends under ${user.budget_max}/night."
                print("\nLLM Response:")
                print(llm.generate_travel_blurb(prompt))
            else:
                print("No user selected.")
        elif choice == '6':
            break
        else:
            print("Invalid option.")


if __name__ == '__main__':
    main()
