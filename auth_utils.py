import bcrypt # type: ignore
import yaml
import os
import streamlit as st

AUTH_CONFIG_PATH = "auth_config.yaml"

def load_auth_config():
    if not os.path.exists(AUTH_CONFIG_PATH):
        return {"credentials": {"usernames": {}}}
    
    with open(AUTH_CONFIG_PATH, 'r') as file:
        return yaml.safe_load(file)

def save_auth_config(config):
    with open(AUTH_CONFIG_PATH, 'w') as file:
        yaml.dump(config, file)

def register_user(username, password, email=None, name=None):
    config = load_auth_config()
    
    if username in config['credentials']['usernames']:
        return False, "Username already exists"
    
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    
    config['credentials']['usernames'][username] = {
        'email': email,
        'name': name,
        'password': hashed
    }
    
    save_auth_config(config)
    return True, "Registration successful"

def verify_user(username, password):
    config = load_auth_config()
    
    if username not in config['credentials']['usernames']:
        return False, "Invalid username"
    
    stored_hash = config['credentials']['usernames'][username]['password']
    if bcrypt.checkpw(password.encode(), stored_hash.encode()):
        return True, "Login successful"
    return False, "Invalid password"

def check_auth_status():
    return 'user' in st.session_state and st.session_state.user is not None

def logout():
    st.session_state.pop('user', None)
    st.session_state.messages = []