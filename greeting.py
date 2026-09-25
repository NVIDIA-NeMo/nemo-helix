def greeting(name):
    name = name.strip()
    if not name:
        raise ValueError("name must not be blank")
    return f"Hello, {name}!"
