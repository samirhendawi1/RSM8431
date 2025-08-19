import os, csv, secrets, re
from dataclasses import dataclass
from typing import Optional

USERS_CSV = "data/users.csv"


def _hash_pw(password: str, salt: str) -> str:
    import hashlib
    h = hashlib.sha256()
    h.update((salt + ":" + password).encode("utf-8"))
    return h.hexdigest()


def _validate_password(pw: str, username: str) -> tuple[bool, str]:
    """
    Rules:
      - >= 8 chars
      - at least 1 lowercase, 1 uppercase, 1 digit, 1 special [^A-Za-z0-9]
      - no spaces
      - must not contain username (case-insensitive) if username is >= 3 chars
    Returns (ok, message). message empty on success.
    """
    if not isinstance(pw, str) or not pw:
        return False, "Password is required."
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if " " in pw:
        return False, "Password must not contain spaces."
    if not re.search(r"[a-z]", pw):
        return False, "Password must include a lowercase letter."
    if not re.search(r"[A-Z]", pw):
        return False, "Password must include an uppercase letter."
    if not re.search(r"[0-9]", pw):
        return False, "Password must include a digit."
    if not re.search(r"[^A-Za-z0-9]", pw):
        return False, "Password must include a special character."
    u = (username or "").strip().lower()
    if len(u) >= 3 and u in pw.lower():
        return False, "Password must not contain your username."
    return True, ""


@dataclass
class User:
    username: str
    # Profile fields remain optional/editable later (kept for recomender defaults)
    name: str = ""
    group_size: int = 0
    environment: str = ""
    budget_min: float = 0.0
    budget_max: float = 0.0


class UserManager:
    """
    Minimal sign-up/sign-in with password requirements and CSV persistence.
    CSV columns: username,password_hash,salt,name,group_size,environment,budget_min,budget_max
    """
    def __init__(self, csv_file: str = USERS_CSV):
        self.csv_file = csv_file
        self.users: dict[str, dict] = {}
        self.current_username: Optional[str] = None
        self._load()

    # ------------- persistence -------------
    def _ensure_dir(self):
        parent = os.path.dirname(self.csv_file)
        if parent and parent not in ("", "."):
            os.makedirs(parent, exist_ok=True)

    def _load(self):
        try:
            with open(self.csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if "username" in r:
                        self.users[r["username"]] = r
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Warning: failed to load users from {self.csv_file}: {e}")

    def _save(self):
        try:
            self._ensure_dir()
            cols = ["username","password_hash","salt","name","group_size","environment","budget_min","budget_max"]
            with open(self.csv_file, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=cols)
                w.writeheader()
                for u in self.users.values():
                    w.writerow({k: u.get(k, "") for k in cols})
        except Exception as e:
            print(f"Error: failed to save users to {self.csv_file}: {e}")

    # ------------- session -------------
    def sign_up(self):
        print("\n-- Sign up --")
        username = input("Choose a username: ").strip().lower()
        if not username:
            print("Username required."); return
        if username in self.users:
            print("That username already exists."); return

        pw1 = input("Create a password: ").strip()
        ok, msg = _validate_password(pw1, username)
        if not ok:
            print(f"Invalid password: {msg}")
            return
        pw2 = input("Confirm password: ").strip()
        if pw1 != pw2:
            print("Passwords did not match."); return

        try:
            salt = secrets.token_hex(16)
            self.users[username] = {
                "username": username,
                "password_hash": _hash_pw(pw1, salt),
                "salt": salt,
                "name": "",
                "group_size": "0",
                "environment": "",
                "budget_min": "0",
                "budget_max": "0",
            }
            self._save()
            self.current_username = username
            print(f"User '{username}' created and signed in.")
        except Exception as e:
            print(f"Error creating user: {e}")

    def sign_in(self):
        print("\n-- Sign in --")
        username = input("Username: ").strip().lower()
        pw = input("Password: ").strip()
        rec = self.users.get(username)
        if not rec:
            print("No such user."); return
        try:
            if rec.get("password_hash") and _hash_pw(pw, rec.get("salt","")) == rec["password_hash"]:
                self.current_username = username
                print(f"Signed in as '{username}'.")
            else:
                print("Invalid credentials.")
        except Exception as e:
            print(f"Error while verifying credentials: {e}")

    def sign_out(self):
        if self.current_username:
            print(f"Signed out '{self.current_username}'.")
            self.current_username = None
        else:
            print("No user is currently signed in.")

    # ------------- profile -------------
    def get_current_user(self) -> Optional[User]:
        if not self.current_username: return None
        r = self.users.get(self.current_username)
        if not r: return None
        try:
            return User(
                username=r["username"],
                name=r.get("name",""),
                group_size=int(float(r.get("group_size",0) or 0)),
                environment=r.get("environment",""),
                budget_min=float(r.get("budget_min",0) or 0),
                budget_max=float(r.get("budget_max",0) or 0),
            )
        except Exception:
            # Return a minimal user if parsing fails
            return User(username=r.get("username",""))

    def edit_profile(self):
        u = self.get_current_user()
        if not u: print("Sign in first."); return
        print("\n-- Edit profile -- (blank keeps current)")
        name = input(f"Name [{u.name}]: ").strip() or u.name
        gs  = input(f"Default group size [{u.group_size}]: ").strip()
        env = input(f"Default environment [{u.environment}]: ").strip().lower() or u.environment
        bmin = input(f"Default budget min [{u.budget_min}]: ").strip()
        bmax = input(f"Default budget max [{u.budget_max}]: ").strip()

        try: group_size = int(gs) if gs else u.group_size
        except: group_size = u.group_size
        try: budget_min = float(bmin) if bmin else u.budget_min
        except: budget_min = u.budget_min
        try: budget_max = float(bmax) if bmax else u.budget_max
        except: budget_max = u.budget_max
        if budget_min > budget_max:
            print("Budget min cannot exceed budget max."); return

        try:
            rec = self.users[u.username]
            rec["name"] = name
            rec["group_size"] = str(group_size)
            rec["environment"] = env
            rec["budget_min"] = str(budget_min)
            rec["budget_max"] = str(budget_max)
            self._save()
            print("Profile updated.")
        except Exception as e:
            print(f"Error updating profile: {e}")

    def view_profile(self):
        """
        Show ONLY the username (per your requirement).
        The caller (main.py) will handle printing last 5 recommendations.
        """
        u = self.get_current_user()
        if not u:
            print("No user signed in."); return
        print("\n--- Profile ---")
        print(f"Username: {u.username}")
        print("----------------\n")

    def delete_user(self):
        username = input("Username to delete: ").strip().lower()
        if username not in self.users:
            print("No such user."); return
        confirm = input("Type DELETE to remove (or 'cancel'): ").strip()
        if confirm != "DELETE":
            print("Cancelled."); return
        try:
            del self.users[username]
            if self.current_username == username:
                self.current_username = None
            self._save()
            print(f"Deleted '{username}'.")
        except Exception as e:
            print(f"Error deleting user: {e}")
