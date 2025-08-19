# UserManager.py

import csv
import os
import sys
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


DATA_DIR = Path("data")
USERS_CSV = DATA_DIR / "users.csv"


@dataclass
class User:
    """Lightweight user object consumed by main/recommender."""
    username: str
    name: str = ""           # first name (for blurbs)
    group_size: int = 0      # not persisted
    environment: str = ""    # not persisted
    budget_min: float = 0.0  # not persisted
    budget_max: float = 0.0  # not persisted


def _hash_pw(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def _validate_password(pw: str, username: str) -> Tuple[bool, str]:
    """
    Rules:
      - >= 8 chars
      - contains uppercase, lowercase, digit, and special char
      - must not contain the username (case-insensitive)
    """
    if len(pw) < 8:
        return False, "Password must be at least 8 characters."
    if username and username.lower() in pw.lower():
        return False, "Password must not contain your username."
    has_upper = any(c.isupper() for c in pw)
    has_lower = any(c.islower() for c in pw)
    has_digit = any(c.isdigit() for c in pw)
    has_special = any(not c.isalnum() for c in pw)
    if not (has_upper and has_lower and has_digit and has_special):
        return False, "Include upper/lowercase letters, a digit, and a special character."
    return True, ""


class UserManager:
    """
    Persisted fields: username, first_name, salt, password_hash
    Session-only fields (never persisted): environment, group_size, budget_min, budget_max
    """
    def __init__(self):
        self.users: Dict[str, Dict[str, str]] = {}
        self.current_username: Optional[str] = None
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self._load()

    # ---------- Persistence ----------

    def _load(self) -> None:
        self.users.clear()
        if not USERS_CSV.exists():
            return
        try:
            with open(USERS_CSV, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    username = (row.get("username") or "").strip().lower()
                    if not username:
                        continue
                    self.users[username] = {
                        "username": username,
                        "first_name": (row.get("first_name") or "").strip(),
                        "salt": (row.get("salt") or "").strip(),
                        "password_hash": (row.get("password_hash") or "").strip(),
                    }
        except Exception as e:
            print(f"[ERROR] Failed to load users: {e}", file=sys.stderr)

    def _save(self) -> None:
        tmp = USERS_CSV.with_suffix(".tmp")
        try:
            with open(tmp, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["username", "first_name", "salt", "password_hash"])
                writer.writeheader()
                for rec in self.users.values():
                    writer.writerow({
                        "username": rec.get("username", ""),
                        "first_name": rec.get("first_name", ""),
                        "salt": rec.get("salt", ""),
                        "password_hash": rec.get("password_hash", ""),
                    })
            os.replace(tmp, USERS_CSV)
        except Exception as e:
            print(f"[ERROR] Failed to save users: {e}", file=sys.stderr)
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    # ---------- Session helpers ----------

    def get_current_user(self) -> Optional[User]:
        if not self.current_username:
            return None
        rec = self.users.get(self.current_username)
        if not rec:
            return None
        return User(
            username=rec.get("username", ""),
            name=rec.get("first_name", ""),
            # the following are intentionally neutral; not persisted
            group_size=0,
            environment="",
            budget_min=0.0,
            budget_max=0.0,
        )

    # ---------- UI actions ----------

    def sign_up(self) -> None:
        print("\n-- Sign up --")
        username = input("Choose a username: ").strip().lower()
        if not username:
            print("Username cannot be empty.")
            return
        if username in self.users:
            print("That username already exists.")
            return

        first_name = input("First name (optional): ").strip()

        pw1 = input("Create a password: ").strip()
        ok, msg = _validate_password(pw1, username)
        if not ok:
            print(f"Invalid password: {msg}")
            return
        pw2 = input("Confirm password: ").strip()
        if pw1 != pw2:
            print("Passwords did not match.")
            return

        try:
            import secrets
            salt = secrets.token_hex(16)
            self.users[username] = {
                "username": username,
                "first_name": first_name,
                "salt": salt,
                "password_hash": _hash_pw(pw1, salt),
            }
            self.current_username = username
            self._save()
            print(f"User '{username}' created and signed in.")
        except Exception as e:
            print(f"[ERROR] Sign up failed: {e}")

    def sign_in(self) -> None:
        print("\n-- Sign in --")
        username = input("Username: ").strip().lower()
        rec = self.users.get(username)
        if not rec:
            print("No such user.")
            return
        pw = input("Password: ").strip()
        try:
            if _hash_pw(pw, rec.get("salt", "")) != rec.get("password_hash", ""):
                print("Incorrect password.")
                return
        except Exception as e:
            print(f"[ERROR] Authentication error: {e}")
            return
        self.current_username = username
        print(f"Signed in as '{username}'.")

    def sign_out(self) -> None:
        if self.current_username:
            print(f"Signed out '{self.current_username}'.")
        self.current_username = None

    def view_profile(self) -> None:
        u = self.get_current_user()
        if not u:
            print("Sign in first.")
            return
        print("\n--- Profile ---")
        print(f"Username: {u.username}")
        if u.name:
            print(f"First name: {u.name}")
        print("----------------")

    def edit_profile(self) -> None:
        """
        Restrict edits to:
          - First name (optional)
          - Username (optional)
          - Password (optional; requires current password)
        """
        u = self.get_current_user()
        if not u:
            print("Sign in first.")
            return

        print("\n-- Edit profile (first name, username, password) --")
        rec = self.users.get(self.current_username)
        if not rec:
            print("Internal error: current user record not found.")
            return

        # First name
        new_first = input(f"First name (blank to keep '{rec.get('first_name','')}'): ").strip()
        if new_first != "":
            rec["first_name"] = new_first

        # Username
        new_username = input(f"New username (blank to keep '{u.username}'): ").strip().lower()
        if new_username and new_username != u.username:
            if new_username in self.users:
                print("That username already exists.")
                return
            try:
                # move record under new key
                self.users.pop(u.username, None)
                rec["username"] = new_username
                self.users[new_username] = rec
                self.current_username = new_username
                print(f"Username changed to '{new_username}'.")
            except Exception as e:
                print(f"[ERROR] Error changing username: {e}")
                return

        # Password
        change_pw = input("Change password? (y/N): ").strip().lower() == "y"
        if change_pw:
            current_pw = input("Current password: ").strip()
            try:
                if _hash_pw(current_pw, rec.get("salt", "")) != rec.get("password_hash", ""):
                    print("Current password is incorrect.")
                    return
            except Exception as e:
                print(f"[ERROR] Error verifying current password: {e}")
                return

            new_pw1 = input("New password: ").strip()
            ok, msg = _validate_password(new_pw1, self.current_username or rec.get("username", ""))
            if not ok:
                print(f"Invalid password: {msg}")
                return
            new_pw2 = input("Confirm new password: ").strip()
            if new_pw1 != new_pw2:
                print("Passwords did not match.")
                return

            try:
                import secrets
                salt = secrets.token_hex(16)
                rec["salt"] = salt
                rec["password_hash"] = _hash_pw(new_pw1, salt)
                print("Password updated.")
            except Exception as e:
                print(f"[ERROR] Error updating password: {e}")
                return

        # Persist any edits
        self._save()

    def delete_user(self) -> None:
        u = self.get_current_user()
        if not u:
            print("Sign in first.")
            return
        sure = input(f"Type your username '{u.username}' to confirm deletion: ").strip().lower()
        if sure != u.username:
            print("Cancelled.")
            return
        try:
            self.users.pop(u.username, None)
            self.current_username = None
            self._save()
            print("User deleted.")
        except Exception as e:
            print(f"[ERROR] Failed to delete user: {e}")
